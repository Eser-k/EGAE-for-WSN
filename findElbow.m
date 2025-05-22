function k_opt = findElbow(inertias)

    Kmax = numel(inertias);
    k = 1:Kmax;

    % Normalize k to [0,1] range
    x_norm = (k - k(1)) / (k(end) - k(1));

    % Normalize inertia values to [0,1]
    y_norm = (inertias - inertias(1)) / (inertias(end) - inertias(1));

    % Compute the “distance” from the straight line y = x
    % at each normalized k
    d = y_norm - x_norm;

    % Identify indices of local maxima in the difference curve d:
    % a local maximum at d(i) satisfies d(i)>d(i-1) and d(i)>d(i+1)
    peakIdx = find( ...
        d(2:end-1) > d(1:end-2) & ...
        d(2:end-1) > d(3:end)    ) + 1;

    if isempty(peakIdx)
        k_opt = 10;
        return;
    end

    % Among the detected peaks, choose the one with the largest deviation
    [~, best] = max(d(peakIdx));

    % Map back to the corresponding cluster count
    k_opt = k(peakIdx(best));
end