function [Area,Model]=setParameters(n)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

%%%%%%%%%%%%%%%%%%%%%%%%% Set Inital Parameters %%%%%%%%%%%%%%%%%%%%%%%%

% Field Dimensions - x and y maximum (in meters)
Area.x=n;
Area.y=n;

% Sink Motion pattern 
Sinkx=0.5*Area.x;
Sinky=Sinkx;

%%%%%%%%%%%%%%%%%%%%%%%%% Energy Model (all values in Joules)%%%%%%%%%%%

% Initial energy per node (battery capacity in Joules) 
Eo=0.5;

ETX=50*0.000000001;     % Energy per bit for transmitter electronics
ERX=50*0.000000001;     % Energy per bit for receiver electronics

% Transmit Amplifier types
Efs=10*0.000000000001;     % Free-space (line-of-sight) amplifier energy coefficient
Emp=0.0013*0.000000000001; % Multi-path amplifier energy coefficient 

% Energy per bit for data aggregation at cluster head
EDA=5*0.000000001;   

% Threshold distance to switch between free-space and multipath models
do=sqrt(Efs/Emp);   

%%%%%%%%%%%%%%%%%%%%%%%%% Run Time Parameters %%%%%%%%%%%%%%%%%%%%%%%%%

% maximum number of rounds
rmax=5000;

% Size of each data packet payload in bits sent during steady-state phase
DpacketLen=4000; 

% Hello packet size
HpacketLen=100;

% Number of Packets sended in steady-state phase
NumPacket=1;

% Redio Range
RR=0.5*Area.x*sqrt(2);
%%%%%%%%%%%%%%%%%%%%%%%%% END OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% Save in Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model.n=n;
Model.Areax = Area.x;
Model.Areay = Area.y;
Model.Sinkx=Sinkx;
Model.Sinky=Sinky;
Model.Eo=Eo;
Model.ETX=ETX;
Model.ERX=ERX;
Model.Efs=Efs;
Model.Emp=Emp;
Model.EDA=EDA;
Model.do=do;
Model.rmax=rmax;
Model.DpacketLen=DpacketLen;
Model.HpacketLen=HpacketLen;
Model.NumPacket=NumPacket;
Model.RR=RR;

end