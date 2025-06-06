import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

class EGAE(tf.keras.Model):
 
    def __init__(self,
                 X,
                 A,
                 num_clusters,
                 alpha,
                 hidden_dims=None,
                 acts=None,
                 max_epoch=10,
                 max_iter=50,
                 learning_rate=1e-2,
                 coeff_reg=1e-3):
        super().__init__()

        # Input as Tensors
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.adjacency = tf.convert_to_tensor(A, dtype=tf.float32)
        self.alpha = alpha

        # Layer-Architecture
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.hidden_dims = hidden_dims

        # Activation Function
        # None könnte falsch sein -> falls nicht funktioniert dann hier ist möglicher Bug
        if acts is None:
            acts = [tf.nn.relu] * (len(hidden_dims) - 1) + [None]
        self.acts = acts
        assert len(self.acts) == len(self.hidden_dims)

        # Training Hyperparameter
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg

        # Dimensions
        n = tf.shape(self.X)[0]
        d = tf.shape(self.X)[1]
        self.data_size = n
        self.input_dim = d

        # Placeholder
        self.indicator = None
        self.embedding = self.X

        # Initialize weights
        self._build_up()

        self.Laplacian = get_laplacian(A)

        self.num_clusters = num_clusters


    def _build_up(self):
        self.gcn_weights = []
        prev_dim = int(self.input_dim)
        for hidden_dim in self.hidden_dims:
            w = self.add_weight(
                shape=(prev_dim, hidden_dim),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True)
            self.gcn_weights.append(w)
            prev_dim = hidden_dim    

    def call(self, inputs=None, training=False):

        x = self.X

        for i, w in enumerate(self.gcn_weights):
            x = tf.matmul(x, w)
            x = tf.matmul(self.Laplacian, x)
            act = self.acts[i]
            if act is not None:
                x = act(x)

        epsilon = 1e-7
        norm = tf.norm(x, axis=1, keepdims=True)
        norm = tf.maximum(norm, epsilon)
        z = x / norm
        self.embedding = z

        recons_A = tf.matmul(z, z, transpose_b=True)
        return recons_A       

    def build_loss_reg(self):
        loss = 0.0
        for w in self.gcn_weights:
            loss += tf.reduce_sum(tf.abs(w))  
        return loss

    def build_loss(self, recons_A):
        
        diag_vals = tf.linalg.diag_part(recons_A)
        recons_A_no_diag = recons_A - tf.linalg.diag(diag_vals)

        N = tf.cast(self.data_size, tf.float32)
        E_pos = tf.reduce_sum(self.adjacency)
        pos_weight = (N * N - E_pos) / E_pos
        eps = 1e-7
        
        term_pos = pos_weight * self.adjacency * tf.math.log(
            tf.math.maximum(eps, recons_A_no_diag)**-1
        )
        term_neg = (1.0 - self.adjacency) * tf.math.log(
            tf.math.maximum(eps, 1.0 - recons_A_no_diag)**-1
        )
        loss_1 = tf.reduce_sum(term_pos + term_neg) / (N * N)

        Zt = tf.transpose(self.embedding)                         # (d, n)
        P = self.indicator                                       # (n, k)
        proj = tf.matmul(tf.matmul(Zt, P), tf.transpose(P))      # (d, n)
        diff = Zt - proj
        
        loss_2 = tf.norm(diff, ord='fro', axis=[0,1])**2 / tf.cast(tf.size(diff), tf.float32)

        loss_reg = self.build_loss_reg()
        
        return loss_1 + self.alpha * loss_2 + self.coeff_reg * loss_reg

    def update_indicator(self):
        
        Z = tf.stop_gradient(self.embedding)  # Form (n, d)

        s, U, V = tf.linalg.svd(Z, full_matrices=False, compute_uv=True)

        P = U[:, :self.num_clusters] 

        self.indicator = tf.stop_gradient(P)

    def clustering(self):
        
        eps = 1e-7
        norms = tf.norm(self.indicator, axis=1, keepdims=True)      # (n,1)
        norms = tf.maximum(norms, eps)                                
        indicator_norm = self.indicator / norms                     # (n,c)

        data = indicator_norm.numpy()                               # (n,c)

        km = KMeans(n_clusters=self.num_clusters, n_init=10).fit(data)
        labels = km.labels_                                           

        return labels

    def run(self):

        self.update_indicator()

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        objs = []

        for epoch in range(self.max_epoch):
            for it in range(self.max_iter):
                with tf.GradientTape() as tape:
                    recons_A = self(training=True)
                    loss = self.build_loss(recons_A)

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                objs.append(loss.numpy())

            self.update_indicator()
            labels = self.clustering()  
            print(f"Epoch {epoch:2d} — loss: {loss.numpy():.4f} — clusters: {np.unique(labels)}")

        return np.array(objs)

    def build_pretrain_loss(self, recons_A):
    
        diag_vals    = tf.linalg.diag_part(recons_A)
        recons_no_diag = recons_A - tf.linalg.diag(diag_vals)
        
        N = tf.cast(self.data_size, tf.float32)
        E_pos = tf.reduce_sum(self.adjacency)
        pos_weight = (N * N - E_pos) / E_pos
        eps = 1e-7
        
        term_pos = pos_weight * self.adjacency * tf.math.log(tf.math.maximum(recons_no_diag, eps)**-1)
        term_neg = (1.0 - self.adjacency) * tf.math.log(tf.math.maximum(1.0 - recons_no_diag, eps)**-1)
        recon_loss = tf.reduce_sum(term_pos + term_neg) / (N * N)
        
        reg_loss = self.build_loss_reg()
    
        return recon_loss + self.coeff_reg * reg_loss


    def pretrain(self, pretrain_steps, learning_rate=None):
        lr = self.learning_rate if learning_rate is None else learning_rate
        optimizer = tf.keras.optimizers.Adam(lr)
        
        for step in range(pretrain_steps):
            with tf.GradientTape() as tape:
                recons_A = self(training=True)
                loss = self.build_pretrain_loss(recons_A)
            
            grads = tape.gradient(loss, self.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))

            print(f"Pretrain step {step+1}/{pretrain_steps} — loss: {loss.numpy():.4f}")


def get_laplacian(A):
    
    A = tf.convert_to_tensor(A, dtype=tf.float32)
    n = tf.shape(A)[0]
    I = tf.eye(n, dtype=tf.float32)
    
    L = A + I
    D = tf.reduce_sum(L, axis=1)                         
    D_inv_sqrt = tf.linalg.diag(tf.math.pow(D, -0.5))    
    
    Laplacian = D_inv_sqrt @ L @ D_inv_sqrt
    return Laplacian


def test_egae():
    # --- 1) Kleines Beispiel-Daten-Setup ---
    n = 6       # 6 Knoten
    d = 4       # 4 Features
    k = 2       # 2 Cluster
    
    # Zufällige Features
    X = np.random.randn(n, d).astype('float32')
    
    # Einfache Adjazenz: zwei Cliquen à 3 Knoten
    A = np.zeros((n, n), dtype='float32')
    A[0:3, 0:3] = 1
    A[3:6, 3:6] = 1
    # symmetrisch machen und Self-Loops via get_laplacian
    A = (A + A.T) / 2
    
    # --- 2) Modell initialisieren ---
    model = EGAE(
        X, A,
        num_clusters=k,
        alpha=1e-2,
        hidden_dims=[8, 4],
        max_epoch=3,
        max_iter=5,
        learning_rate=1e-2,
        coeff_reg=1e-3
    )
    
    # --- 3) Pretraining ---
    model.pretrain(pretrain_steps=20)
    
    # --- 4) Gemeinsames Training ---
    losses = model.run()
    print("Loss history shape:", losses.shape)
    
    # --- 5) Clustering auslesen ---
    model.update_indicator()
    labels = model.clustering()
    print("Predicted cluster labels:", labels)
    print("Einzigartige Labels:", np.unique(labels))
    
    # Kurze Plausibilitätsprüfung:
    # Wir erwarten im Toy-Beispiel genau 2 verschiedene Labels.
    assert len(np.unique(labels)) == k, "Anzahl der Cluster stimmt nicht!"
    print("Test erfolgreich durchgelaufen ✅")


if __name__ == "__main__":
    # Wenn Du die Datei direkt aufrufst, wird der Test ausgeführt.
    test_egae()