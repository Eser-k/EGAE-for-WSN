function [CH, Sensors] = selectCH(Sensors, Model, cluster_labels, fis)
    % selectCH: Wählt pro Cluster den Sensorkopf mit dem höchsten Fuzzy-Score
    % Inputs:
    %   Sensors         – Array mit Feldern .E, .xd, .yd
    %   Model           – Struktur, enthält .n und Optionales: .commRange
    %   cluster_labels  – n×1 Vektor mit Cluster-IDs 1…K für jeden Sensor
    %   fis             – geladenes Fuzzy-Inferenzsystem (Mamdani)
    %
    % Outputs:
    %   CH              – Struktur mit den IDs der gewählten Clusterheads
    %   Sensors         – wie Eingabe, mit .type='C' für CHs
    
    n = Model.n;
    coords = [[Sensors(1:n).xd]' [Sensors(1:n).yd]'];  % n×2 Positionsmatrix
    energyVec  = [Sensors(1:n).E]';
    unique_labels = unique(cluster_labels(:));
    CH = struct('id',{});
    
    % globale Normalisierung der Energie
    E_min = min(energyVec);
    E_max = max(energyVec);
    
    for ci = 1:numel(unique_labels)
        c = unique_labels(ci);
        members = find(cluster_labels==c & energyVec>0);
        if isempty(members), continue; end
        
        % Vorbereiten der Descriptoren
        % 1) Energie
        e_norm = (energyVec(members)-E_min)/(E_max-E_min+eps);
        
        % 2) Konzentration: Anzahl der anderen Mitglieder im Quadrat-Umkreis
        %    (hier 20×20 m, d.h. +/-10 m in x/y)
        X = coords(members,1); Y = coords(members,2);
        conc = zeros(size(members));
        for k = 1:numel(members)
            dx = abs(coords(:,1)-X(k));
            dy = abs(coords(:,2)-Y(k));
            conc(k) = sum(dx<=30 & dy<=30) - 1;  % ohne sich selbst
        end
        % normieren auf [0,1]
        conc_norm = (conc - min(conc))/(max(conc)-min(conc)+eps);
        
        % 3) Zentralität: Summe der quadrierten Distanzen zu allen anderen
        cent = zeros(size(members));
        for k = 1:numel(members)
            d = sqrt((coords(:,1)-X(k)).^2 + (coords(:,2)-Y(k)).^2);
            cent(k) = sum(d(members).^2);  % nur im Cluster
        end
        % invertieren, damit niedrige Quadriersumme → hoher Zentralitätswert
        cent_inv = max(cent) - cent;
        cent_norm = (cent_inv - min(cent_inv))/(max(cent_inv)-min(cent_inv)+eps);
        
        % 4) Fuzzy-Scoring aller Kandidaten
        scores = zeros(size(members));
        for k = 1:numel(members)
            inputVec = [e_norm(k), conc_norm(k), cent_norm(k)];
            scores(k) = evalfis(inputVec, fis);
        end
        
        % 5) Wahl des Besten
        [~, idxBest] = max(scores);
        bestNode = members(idxBest);
        
        CH(end+1).id      = bestNode;   %#ok<AGROW>
        Sensors(bestNode).type = 'C';
    end
end