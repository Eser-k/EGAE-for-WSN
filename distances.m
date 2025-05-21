function d = distances(X)

%% Developed by Eser Kayali
 % eser-k@web.de

 %  Computes the Euclidean distances for all unique pairs of points.
 %
 %  This function takes an [n × 2] or [n × k] matrix X, 
 %  where each row is the coordinates of a point 
 %  (e.g., cluster‐head positions), and returns a column vector `d` of 
 %  length n*(n–1)/2 containing the pairwise distances between
 %  every distinct pair (i, j) with i < j. This is useful for proximity 
 %  checks without duplicating symmetric pairs or self‐distances.

    [n, ~] = size(X);
    numPairs = n*(n-1)/2;
    d = zeros(numPairs, 1);
    idx = 1;
    for i = 1:n-1
        for j = i+1:n
            diff = X(i,:) - X(j,:);
            d(idx) = sqrt(sum(diff.^2));
            idx = idx + 1;
        end
    end
end