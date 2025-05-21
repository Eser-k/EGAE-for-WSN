function Sensors=JoinToNearestCH(Sensors,Model,TotalCH)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Assigns each alive sensor node to the nearest cluster head (CH) 
 % if within communication range; otherwise falls back to 
 % direct sink communication.
 %
 %  Steps:
 %    1. Compute Euclidean distances from each sensor to each CH.
 %    2. For each sensor, find the minimum distance and
 %       corresponding CH index.
 %    3. If the sensor is alive (energy > 0) and:
 %         • the nearest CH is within radio range (Model.RR), and
 %         • that distance is less than the sensor’s distance to the sink,
 %       then set MCH to that CH and record dis2ch accordingly.
 %    4. Otherwise, set MCH to the sink (node n+1) and dis2ch to dis2sink.

n=Model.n;
m=length(TotalCH);

if(m>=1)
    % Preallocate distance matrix: rows=CHs, cols=sensors
    D=zeros(m,n); 

    % Compute distance from each sensor i to each CH j
    for i=1:n     
        for j=1:m
            
            D(j,i)=sqrt((Sensors(i).xd-Sensors(TotalCH(j).id).xd)^2+ ...
                (Sensors(i).yd-Sensors(TotalCH(j).id).yd)^2);        
        end   
    end 
    
    % For each sensor, find the nearest CH distance and index
    [Dmin,idx]=min(D, [], 1);

    % Assign each sensor to nearest CH or to sink if criteria not met
    for i=1:n       
        if (Sensors(i).E>0)

            % If alive and nearest CH is within radio range
            % and closer than sink, join that CH
            if (Dmin(i) <= Model.RR && Dmin(i)<Sensors(i).dis2sink )
                Sensors(i).MCH=TotalCH(idx(i)).id;
                Sensors(i).dis2ch=Dmin(i);
            else
                % Otherwise set MCH to sink (node n+1)
                Sensors(i).MCH=n+1;
                Sensors(i).dis2ch=Sensors(i).dis2sink;
            end
        end
        
    end 
end

end

