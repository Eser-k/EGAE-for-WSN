function A = createAdjacencyMatrix(Sensors, Model)
%% Developed by Eser Kayali
 % eser-k@web.de

 % createAdjacencyMatrix: Build the graph adjacency matrix 
 %  for the sensor network.
 %  A = createAdjacencyMatrix(Sensors, Model) returns an n×n binary matrix A
 %  where A(i,j) = 1 if sensors i and j are within adjacency range, and 0
 %  otherwise.

    % number of sensor nodes
    n = Model.n;  

    adjacencyDistance = 20;        

    % allocate n×n matrix
    A = zeros(n, n);     

    % Compute pairwise connectivity
    for i = 1:n
        for j = 1:n
            if i == j
                % leave diagonal zero; 
                % self-loops handled in egae.py
                A(i,j) = 0;
            else
                % Euclidean distance between node i and j
                distance = sqrt((Sensors(i).xd - Sensors(j).xd)^2 + ...
                                (Sensors(i).yd - Sensors(j).yd)^2);
                % mark edge if within range
                if distance <= adjacencyDistance
                    A(i,j) = 1;
                else
                    A(i,j) = 0;
                end
            end
        end
    end
end