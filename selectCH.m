function [CH, Sensors] = selectCH(Sensors, Model, cluster_labels, fis)
    
    n         = Model.n;
    coords    = [[Sensors(1:n).xd]' [Sensors(1:n).yd]'];
    energyVec = [Sensors(1:n).E]';
    labels    = unique(cluster_labels(:));
    CH        = struct('id',{});
    
    E_min = min(energyVec);
    E_max = max(energyVec);
    
    lowEnergyThresh = 0.15;  
    
    for li = 1:numel(labels)

        c = labels(li);

        members = find(cluster_labels==c & energyVec>0);
        if isempty(members)
            continue
        end
        
        if E_min < lowEnergyThresh * Model.Eo 
            [~, localIdx] = max(energyVec(members));
            bestNode      = members(localIdx);
            
        else

            e_norm = (energyVec(members)-E_min)/(E_max-E_min+eps);

            X = coords(members,1); 
            Y = coords(members,2);
            conc = zeros(size(members));
            
            for k = 1:numel(members)
                dx = abs(coords(:,1)-X(k));
                dy = abs(coords(:,2)-Y(k));
                conc(k) = sum(dx<=30 & dy<=30)-1;
            end

            conc_norm = (conc-min(conc))/(max(conc)-min(conc)+eps);
            
            cent = zeros(size(members));

            for k = 1:numel(members)
                d = sqrt((coords(:,1)-X(k)).^2+(coords(:,2)-Y(k)).^2);
                cent(k) = sum(d(members).^2);
            end

            cent_norm = (cent - min(cent)) / (max(cent) - min(cent) + eps);
            
            scores = zeros(size(members));

            for k = 1:numel(members)
                inputVec = [e_norm(k), conc_norm(k), cent_norm(k)];
                scores(k) = evalfis(inputVec, fis);
            end

            [~, bestLocal] = max(scores);
            bestNode = members(bestLocal);
        end
        
        CH(end+1).id = bestNode;  %#ok
        Sensors(bestNode).type = 'C';
    end
end