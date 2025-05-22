function inertias = computeInertia(Z, Kmax)

  inertias = zeros(1, Kmax);

  % Loop over possible numbers of clusters
  for k = 1:Kmax

    % Create a Python scikit-learn KMeans instance with:
        %   - k clusters
        %   - 5 random initializations (n_init)
        %   - up to 200 iterations per initialization (max_iter)
        %   - fixed random seed for reproducibility  
    km = py.sklearn.cluster.KMeans( ...
      pyargs(...
        'n_clusters', int32(k), ...
        'n_init',     int32(5), ...
        'max_iter',   int32(200), ...
        'random_state', int32(0) ...
      ) ...
    );

    % Convert Z to a numpy array and run the kMeans algorithm
    km = km.fit(py.numpy.array(Z));

    % Extract the inertia_ attribute (sum of squared distances to
    % nearest cluster center) and store it in the output vector
    inertias(k) = double(km.inertia_);
  
  end
end