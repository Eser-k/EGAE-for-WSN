function inertias = computeInertia(Z, Kmax)

    inertias = zeros(1, Kmax);
  
    for k = 1:Kmax
      [~, ~, sumD] = kmeans(Z, k, ...
        'Replicates', 5, ...      
        'MaxIter', 300, ...       
        'Display', 'off');        
      inertias(k) = sum(sumD);    
    end
end