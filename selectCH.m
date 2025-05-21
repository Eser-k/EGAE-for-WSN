function [CH, Sensors] = selectCH(Sensors, Model, cluster_labels)

    n = Model.n;
    
    % Build a column vector of each sensor’s remaining energy
    energyVec = [Sensors(1:n).E]';
    
    % Determine which cluster IDs actually appear
    unique_labels = unique(cluster_labels(:));
    
    % Prepare output array of cluster-head sensors
    CH = struct('id',{});
    
    % For each distinct cluster label...
    for i = 1:length(unique_labels)
        c = unique_labels(i);

        % Find indices of alive members in this cluster
        members = find(cluster_labels==c & energyVec>0);

        if isempty(members), continue; end

        % Among these members, pick the one with maximum energy
        [~, idx] = max(energyVec(members));

        % Retrieve the global sensor index of the selected cluster‐head
        ch = members(idx);

        CH(end+1).id = ch; %#ok

        % Mark the sensor at index ch as a cluster‐head
        Sensors(ch).type = 'C';    
    end
    
end