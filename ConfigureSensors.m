function Sensors=ConfigureSensors(Model,n,GX,GY)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Initializes the sensor network structure array with positions, energy,
 %  and default parameters for all nodes including the sink.

%--- Define default template for one sensor node ---%

EmptySensor.xd=0;        % x-position
EmptySensor.yd=0;        % y-position
%EmptySensor.G=0;
EmptySensor.df=0;        % dead flag: 0=alive, 1=dead
EmptySensor.type='N';    % node type: 'N' = normal, 'C' = cluster-head
EmptySensor.E=0;         % remaining energy (Joules)
EmptySensor.id=0;        % unique node identifier
EmptySensor.dis2sink=0;  % distance to sink
EmptySensor.dis2ch=0;    % distance to chosen cluster-head
EmptySensor.MCH=n+1;     % member-of-CH ID, defaults to sink

%--- Preallocate struct array for sensors + sink ---%
Sensors=repmat(EmptySensor,n+1,1);

%--- Configure each actual sensor node ---%
for i=1:1:n
    
    % set x-coordinate
    Sensors(i).xd=GX(i);  
    
    %set y-coordinate
    Sensors(i).yd=GY(i);

    % Determinate whether in previous periods has been clusterhead or not? 
    % not=0 and be=n
    % not needed for GCN
    %Sensors(i).G=0;

    % mark as alive initially
    Sensors(i).df=0; 

    % start as normal node 
    Sensors(i).type='N';

    % assign initial energy
    Sensors(i).E=Model.Eo;

    % assign node ID
    Sensors(i).id=i;

    %Sensors(i).RR=Model.RR;
    
end 

%--- Configure the sink node at index n+1 ---%
Sensors(n+1).xd=Model.Sinkx;     % sink x-position
Sensors(n+1).yd=Model.Sinky;     % sink y-position
Sensors(n+1).E=100;
Sensors(n+1).id=n+1;             % sink ID

end