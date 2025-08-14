function G = createCHGraph(Sensors, Model, TotalCH)

    chIds = [TotalCH.id];
    if ~isempty(chIds)
        chIds = chIds([Sensors(chIds).E] > 0);  % nur lebende CHs
    end

    sinkId  = Model.n + 1;
    nodeIds = [chIds, sinkId];
    m = numel(nodeIds);

    pos = zeros(m,2);
    for i = 1:m
        pos(i,1) = Sensors(nodeIds(i)).xd;
        pos(i,2) = Sensors(nodeIds(i)).yd;
    end

    D = zeros(m,m);
    W = inf(m,m);

    kbits = Model.DpacketLen;
    ERx   = Model.ERX * kbits;

    function e = txCost(d)
        if d > Model.do
            e = Model.ETX*kbits + Model.Emp*kbits*(d^4);
        else
            e = Model.ETX*kbits + Model.Efs*kbits*(d^2);
        end
    end

    sinkIdx = m;  

    for i = 1:max(0,m-1)
        for j = i+1:m
            dx  = pos(i,1) - pos(j,1);
            dy  = pos(i,2) - pos(j,2);
            dij = sqrt(dx*dx + dy*dy);

            D(i,j) = dij;  
            D(j,i) = dij;

            eij = txCost(dij) + ERx;          
            
            W(i,j) = eij;
            W(j,i) = eij;
        end
    end

    G.nodeIds   = nodeIds;
    G.pos       = pos;
    G.W         = W;
    G.D         = D;
    G.sinkIndex = sinkIdx;
    G.chIndices = 1:max(0,m-1);
    G.kbits     = kbits;
end