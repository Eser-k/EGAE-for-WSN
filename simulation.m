%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
% eser-k@web.de

clc;
clear;
close all;
warning off all;

%% Create sensor nodes, Set Parameters and Create Energy Model

%%%%%%%%%%%%%%%% Initial Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=100;                          % Number of Nodes in the field
[Area,Model]=setParameters(n);  % Set Parameters Sensors and Network
    
%%%%%%%%%%%%%%%% Configuration of the Sensors %%%%%%%%%%%%%%%%%%%%%%%%%
    
% Create a random scenario
CreateRandomSen(Model,Area);  

% Load sensor Location
load Locations
    
Sensors=ConfigureSensors(Model,n,X,Y);
    
%%%%%%%%%%%%%%%%% Initialization of the sarameters %%%%%%%%%%%%%%%%%%%%
    
% Flag indicating whether the first node death has occurred 
% (0 = not yet, 1 = yes)
flag_first_dead=0;  

% Total number of sensor nodes that have died so far
deadNum=0;          
    
% Compute the total initial energy of the network
initEnergy=0;       
for i=1:n
      initEnergy=Sensors(i).E+initEnergy;
end
    
% Total number of routing packets sent by all nodes in each round
SRP=zeros(1,Model.rmax);    

% Total number of routing packets received by all nodes in each round
RRP=zeros(1,Model.rmax);   

% Total number of data packets sent (to CHs or sink) in each round
SDP=zeros(1,Model.rmax);    

% Total number of data packets received (at CHs or sink) in each round
RDP=zeros(1,Model.rmax);    

% Number of sensors still alive at the end of each round
AliveSensors= zeros(1,Model.rmax);

% Total remaining energy of all sensors at the end of each round
SumEnergyAllSensor = zeros(1,Model.rmax);

% Average remaining energy per alive sensor at the end of each round
AvgEnergyAllSensor = zeros(1,Model.rmax);

% Total energy consumed by all sensors during each round
RoundEnergy = zeros(1,Model.rmax);

% Energy heterogeneity metric 
% (variance of remaining sensor energies) per round
Enheraf = zeros(1,Model.rmax);

% Cumulative number of dead nodes at the end of each round
Count_DeadNodes=zeros(1,Model.rmax);

% Total remaining energy of all sensors at the end of each round
SumEnergyAllSensor(1) = initEnergy;

% Track number of alive nodes (initially all)
alive = n;
AliveSensors(1)= n;

%%%%%%%%%%%%%%%%%% cluster with kMeans  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
positions = createFeatureMatrix(Sensors,Model);
kmax = int32(n/5);
inertias = computeInertia(positions, kmax);

% Visualize the inertia values
figure('Name','Elbow','NumberTitle','off');
plot(1:kmax, inertias, '-o');
xlabel('k');
ylabel('Inertia');
title('Elbow Curve');
grid on;

% Determine the optimal number of clusters by 
% finding the “elbow” in the inertia curve
k_opt = findElbow(inertias);
fprintf('Optimal number of clusters (Elbow): %d\n', k_opt);

num_clusters = k_opt;
cluster_labels = kmedoids(positions, k_opt);

% Generate a palette of distinct colors (one per cluster) 
cmap = jet(num_clusters);

% Create a new figure window named “Sensor Network” 
simFig = figure('Name','Sensor Network','NumberTitle','off');

% Add a set of axes to the figure for plotting the sensor network
axSim  = axes('Parent',simFig);

% Retain existing plots on these axes when adding new graphics
hold(axSim,'on');

% Add a title to the axes showing the current round and dead‐node count,
% and store the handle for later updates
hTitle = title(axSim, 'Round=0, Dead nodes=0','FontSize',12,'FontWeight','bold');

fis = readfis('CH_Selection.fis');

%%%%%%%%%%%%%%%%%%% Start Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize global counters for packet statistics
global srp rrp sdp rdp

srp=0;    % Total number of routing packets sent by all nodes
rrp=0;    % Total number of routing packets received by sink 
sdp=0;    % Total number of data packets sent by all nodes 
rdp=0;    % Total number of data packets received by sink

% Sink (node n+1) broadcasts a "Hello" message to all sensor nodes
Sender=n+1;     
Receiver=1:n;   % Indices of all sensor nodes
Sensors=SendReceivePackets(Sensors,Model,Sender,'Hello',Receiver);
    
% Each sensor computes its distance to the sink
Sensors=disToSink(Sensors,Model);

% Sender=1:n;     %All nodes
% Receiver=n+1;   %Sink
% Sensors=SendReceivePackets(Sensors,Model,Sender,'Hello',Receiver);

% Record the initial packet counts for round 0
SRP(1)=srp;
RRP(1)=rrp;  
SDP(1)=sdp;
RDP(1)=rdp;
    
%% Main loop program
for r=1:1:Model.rmax

%%%%%%%%%%%%%%%%%%% Initialization for this round %%%%%%%%%%%%%%%%%%%%%
    
    % Reset per-round communication counters
    srp=0;          
    rrp=0;          
    sdp=0;          
    rdp=0;

    % Store the zeroed counters for this round 
    % (will be updated as packets are sent)
    SRP(r+1)=srp;
    RRP(r+1)=rrp;  
    SDP(r+1)=sdp;
    RDP(r+1)=rdp;
        
    % Reset all sensors to their default state before new round:
    %   - Clear any previous CH assignment 
    %   - Mark every node as a normal sensor ('N')
    %   - Set distance to CH to ∞ so the closest CH will overwrite it
    Sensors=resetSensors(Sensors,Model);
    
%%%%%%%%%%%%%%%%%%%%% cluster head election %%%%%%%%%%%%%%%%%%%%%%%%%%% 

    % Use the computed cluster_labels to select one alive sensor per cluster 
    % (the one with highest remaining energy) as the cluster head.
    [TotalCH,Sensors]=selectCH(Sensors,Model,cluster_labels,fis); 
    
%%%%%%%%%%%%%%%%%%%% plot sensors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    % Make sure we’re drawing into the “Sensor Network” figure
    figure(simFig)

    % Activate the specific axes object we created earlier
    axes(axSim); %#ok

    % Clear those axes, removing any existing points or lines
    cla(axSim);                   
    
    % Build a column vector of each sensor’s remaining energy
    energyVec = [Sensors(1:n).E]';

    % Iterate over each cluster
    for c = 1:num_clusters
        
        % Create a boolean mask selecting only those sensors that both 
        % belong to cluster and are still alive
        mask = (cluster_labels==c) & (energyVec>0);
    
        % Find the indices of sensors that both belong to 
        % cluster c and are still alive
        idx  = find(mask);
    
        % If no sensors meet the criteria for this cluster, 
        % skip to the next iteration
        if isempty(idx), continue; end
    
        % Gather the x-positions (xd) of all sensors in the cluster
        xs = [Sensors(idx).xd];

        % Gather the y-positions (yd) of all sensors in the cluster
        ys = [Sensors(idx).yd];
        
        % Plot filled markers at (xs,ys) with size 36, colored by the c-th
        % row of cmap, and black edges
        scatter(xs, ys, 36, cmap(c,:), 'filled', 'MarkerEdgeColor','k');  
    end
    
    % Find indices of dead sensors 
    deadIdx = find(energyVec<=0);

    if ~isempty(deadIdx)
        % Gather the x-positions (xd) of all dead sensors
        xs = [Sensors(deadIdx).xd];

        % Gather the y-positions (yd) of all dead sensors
        ys = [Sensors(deadIdx).yd];

        % Plot dead nodes as small red dots
        scatter(xs, ys, 36, 'r', '.');
    end
    
    % Plot the sink node as a large green star with a thick outline
    sinkX = Sensors(n+1).xd;
    sinkY = Sensors(n+1).yd;
    scatter(sinkX, sinkY, 100, 'g', '*', 'LineWidth',1.5);
    
    % Ensure the axes have equal length units and make the 
    % plot box square (1:1 aspect ratio)
    axis square;

    % Update the figure title to show the current round number 
    % and dead node count
    hTitle.String = sprintf('Round=%d, Dead nodes=%d', r, deadNum);

    drawnow; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Advertise CH role to neighbors within radio range
    for i=1:length(TotalCH)
        Sender=TotalCH(i).id;
        SenderRR=Model.RR;
        Receiver=findReceiver(Sensors,Model,Sender,SenderRR);   
        Sensors=SendReceivePackets(Sensors,Model,Sender,'Hello',Receiver);
    end 
    
    % Sensors join the nearest clusterhead 
    Sensors=JoinToNearestCH(Sensors,Model,TotalCH);

 %%%%%%%%%%%%%%%%%%%% Plot links from non-CH nodes to their cluster 
% heads after setup phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Activate the simulation figure for drawing communication links
    figure(simFig)

    for i=1:n
        % Only draw links for alive normal nodes (‘N’) 
        % whose distance to their cluster head is less 
        % than their distance to the sink
        if (Sensors(i).type=='N' && Sensors(i).dis2ch<Sensors(i).dis2sink && ...
                Sensors(i).E>0)
            
            XL=[Sensors(i).xd ,Sensors(Sensors(i).MCH).xd];
            YL=[Sensors(i).yd ,Sensors(Sensors(i).MCH).yd];
            hold on
            line(XL,YL, 'Color', 'k', 'LineWidth', 0.7);
            
        end   
    end

    % Briefly pause so that each round’s links are visible before updating
    pause(0.1);
    
%%%%%%%%%%%%%%%%%%%%% steady-state phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % In the steady-state phase, each alive sensor sends its data 
    % to its assigned cluster head NumPacket times per round

    NumPacket=Model.NumPacket;
    for i=1:NumPacket 
        % Each sensor sends its data packet to its cluster head
        for j=1:length(TotalCH)
            
            Receiver=TotalCH(j).id;
            Sender=findSender(Sensors,Model,Receiver); 
            Sensors=SendReceivePackets(Sensors,Model,Sender,'Data',Receiver);
            
        end 
    end
    
    % Once each cluster head has aggregated data from its members, 
    % it forwards the consolidated packet directly to the sink
    for i=1:length(TotalCH)
            
        Receiver=n+1;                % ID of the sink (base station)
        Sender=TotalCH(i).id;        % ID of the i-th cluster head 
        Sensors=SendReceivePackets(Sensors,Model,Sender,'Data',Receiver);     
    end
     
    % Any sensor node that is not part of a cluster head’s group 
    % sends its own data packet straight to the sink
    for i=1:n
        if(Sensors(i).MCH==Sensors(n+1).id)
            Receiver=n+1;               % ID of the sink (base station)
            Sender=Sensors(i).id;       % ID of the Other Nodes 
            Sensors=SendReceivePackets(Sensors,Model,Sender,'Data',Receiver);
        end
    end
 
%%%%%%%%%%%%%%%%%%%%% Statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Record packet counters for this round
    SRP(r+1)=srp;
    RRP(r+1)=rrp;  
    SDP(r+1)=sdp;
    RDP(r+1)=rdp;
    
    % Compute the number of alive sensors and aggregate their 
    % remaining energy
    alive=0;
    SensorEnergy=0;
    for i=1:n
        if Sensors(i).E>0
            alive=alive+1;
            SensorEnergy=SensorEnergy+Sensors(i).E;
        end
    end
    
    % Store the count of alive sensors in this round
    AliveSensors(r+1) = alive; 
    
    % Store the total remaining energy across all alive sensors
    SumEnergyAllSensor(r+1)=SensorEnergy; 
    
    % Average energy per alive sensor
    AvgEnergyAllSensor(r+1)=SensorEnergy/alive; 
    
    % total energy consumed by all sensors during the current round
    RoundEnergy(r+1)=SumEnergyAllSensor(r) - SumEnergyAllSensor(r+1); 
    
    % Compute the total squared deviation of each alive sensor’s 
    % energy from the round’s mean
    En=0;
    for i=1:n
        if Sensors(i).E>0
            En=En+(Sensors(i).E-AvgEnergyAllSensor(r+1))^2;
        end
    end

    Enheraf(r+1)=En/alive; 

    deadNum = sum(energyVec<=0);
    Count_DeadNodes(r+1) = deadNum;

    % Save r'th period when the first node dies
    if (deadNum>=1)      
        if(flag_first_dead==0)
            first_dead=r;
            flag_first_dead=1;
        end  
    end

    if alive <= n/2 && ~exist('half_dead','var')
        half_dead = r;
    end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

   % If all sensor nodes are dead, record the final round 
   % and exit the loop
   if(n==deadNum)
       lastRound=r;  
       break;
   end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display the round when the first sensor node died
fprintf("First node death:  %d\n", first_dead);

% Display the round when half of the sensor nodes have died
fprintf("Half nodes dead:    %d\n", half_dead);

% Display the total number of rounds until all sensor nodes have died
fprintf("Network Lifetime:   %d\n", lastRound);

% Create an index vector for selecting rounds 
% from 1 up to the last completed round
idx = 1:(lastRound+1);

% Truncate each metrics array to include 
% only the data for the completed rounds
SRP                = SRP(idx);
RRP                = RRP(idx);
SDP                = SDP(idx);
RDP                = RDP(idx);
AliveSensors       = AliveSensors(idx);
SumEnergyAllSensor = SumEnergyAllSensor(idx);
AvgEnergyAllSensor = AvgEnergyAllSensor(idx);
RoundEnergy        = RoundEnergy(idx);
Enheraf            = Enheraf(idx);

% Convert each metric array into a column vector 
SRP               = SRP(:);
RRP               = RRP(:);
SDP               = SDP(:);
RDP               = RDP(:);
AliveSensors      = AliveSensors(:);
SumEnergyAllSensor= SumEnergyAllSensor(:);
AvgEnergyAllSensor= AvgEnergyAllSensor(:);
RoundEnergy       = RoundEnergy(:);
Enheraf           = Enheraf(:);

% Generate a column vector of round indices from 0 up to the last round
Round = (0:lastRound)';   

% Combine all per-round metrics into a single table for further analysis
Stats = table( ...
    Round, ...
    SRP, ...
    RRP, ...
    SDP, ...
    RDP, ...
    AliveSensors, ...
    SumEnergyAllSensor, ...
    AvgEnergyAllSensor, ...
    RoundEnergy, ...
    Enheraf ...
);

% Extract the 'Round' column from the Stats table into a standalone vector
rounds  = Stats.Round;

% Get the names of all metric columns (excluding the first 'Round' column)
metrics = Stats.Properties.VariableNames(2:end);  

% Extract numeric values for all metrics (exclude the first 'Round' column)
data   = Stats{:,2:end};   

% Transpose the data matrix so that metrics become rows 
% and rounds become columns
data_T = data.';      

% Construct a table from the transposed data, using each metric name 
% as a row label and “Round <n>” as the column headers corresponding to 
% each round number
T = array2table( data_T, 'RowNames', metrics, ...
    'VariableNames', compose("Round %d", rounds));

% Export the table to a CSV file
writetable(T, 'stats_by_metric.csv', 'WriteRowNames', true);