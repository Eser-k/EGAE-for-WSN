function inertias = computeInertia(positions, Kmax)
    inertias = zeros(1, Kmax);

    Z = linkage(positions, 'ward');

    for k = 1:Kmax
        labels = cluster(Z, 'maxclust', k);

        totalSS = 0;
        for c = 1:k
            pts = positions(labels == c, :);
            if ~isempty(pts)
                mu = mean(pts, 1);
                diffs = pts - mu;
                totalSS = totalSS + sum(diffs(:).^2);
            end
        end
        inertias(k) = totalSS;
    end
end