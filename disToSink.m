function Sensors=disToSink(Sensors,Model)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Computes and stores the Euclidean distance from 
 %  each sensor to the sink node.
 %
 %  This function iterates over all sensor nodes 
 %  (excluding the sink itself), calculates the straight‐line distance
 %  to the sink (assumed to be the last element in the Sensors array), 
 %  and writes that distance into each sensor’s
 %  `dis2sink` field for later use.

    n=Model.n;
    
    for i=1:n
        
        distance=sqrt((Sensors(i).xd-Sensors(n+1).xd)^2 + ...
            (Sensors(i).yd-Sensors(n+1).yd)^2 );
        
        Sensors(i).dis2sink=distance;
        
    end
    
end