function inertias = computeInertia(positions, Kmax)

    inertias = zeros(1, Kmax);
  
    for k = 1:Kmax
      [~, ~, sumD] = kmedoids(positions, k);     
      inertias(k) = sum(sumD);    
    end
end