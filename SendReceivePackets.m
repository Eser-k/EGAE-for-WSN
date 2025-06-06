function Sensors=SendReceivePackets(Sensors,Model,Sender,PacketType,Receiver)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

%  Simulates sending and receiving packets between sensor nodes,
%  updates each node's energy based on transmission/reception costs,
%  and accumulates global packet statistics.
    
   % Access global counters for packets 
   global srp rrp sdp rdp 

   % Local counters for this call
   sap=0;      % number of successfully sent packets
   rap=0;      % number of successfully received packets


   % Determine packet size based on type
   if (strcmp(PacketType,'Hello'))
       PacketSize=Model.HpacketLen;
   else
       PacketSize=Model.DpacketLen;
   end
   
   %% Transmission energy consumption
    % Loop over each sender and each receiver to deduct transmit energy
   for i=1:length( Sender)
       
      for j=1:length( Receiver)
          

            distance=sqrt((Sensors(Sender(i)).xd-Sensors(Receiver(j)).xd)^2 + ...
               (Sensors(Sender(i)).yd-Sensors(Receiver(j)).yd)^2 );  

            if (distance>Model.do)

                Sensors(Sender(i)).E=Sensors(Sender(i)).E- ...
                    (Model.ETX*PacketSize + Model.Emp*PacketSize*(distance^4));

                % Sent a packet
                if(Sensors(Sender(i)).E>0)
                    sap=sap+1;                 
                end

            else

                Sensors(Sender(i)).E=Sensors(Sender(i)).E- ...
                    (Model.ETX*PacketSize + Model.Efs*PacketSize*(distance^2));

                % Sent a packet
                if(Sensors(Sender(i)).E>0)
                    sap=sap+1;                 
                end

            end
          
      end
      
   end
   
   %% Reception energy consumption
    % Every receiver pays ERX + EDA per packet
   for j=1:length( Receiver)
        Sensors(Receiver(j)).E =Sensors(Receiver(j)).E- ...
            ((Model.ERX + Model.EDA)*PacketSize);
         
   end   
   
   for i=1:length(Sender)
       for j=1:length(Receiver)

            %Received a Packet
            if(Sensors(Sender(i)).E>0 && Sensors(Receiver(j)).E>0)
                rap=rap+1;
            end
       end 
   end
   
    % Update global packet counters
    if (strcmp(PacketType,'Hello'))
        srp=srp+sap;
        rrp=rrp+rap;
    else       
        sdp=sdp+sap;
        rdp=rdp+rap;
    end
   
end

%     else %To Cluster Head
%         
%         for i=1:length( Sender)
%        
%            distance=sqrt((Sensors(Sender(i)).xd-Sensors(Sender(i).MCH).xd)^2 + ...
%                (Sensors(Sender(i)).yd-Sensors(Sender(i).MCH).yd)^2 );   
%        
%            send a packet
%            sap=sap+1;
%            
%            Energy dissipated from Normal sensor
%            if (distance>Model.do)
%            
%                 Sensors(Sender(i)).E=Sensors(Sender(i)).E- ...
%                     (Model.ETX*PacketSize + Model.Emp*PacketSize*(distance^4));
% 
%                 if(Sensors(Sender(i)).E>0)
%                     rap=rap+1;                 
%                 end
%             
%            else
%                 Sensors(Sender(i)).E=Sensors(Sender(i)).E- ...
%                     (Model.ETX*PacketSize + Model.Emp*PacketSize*(distance^2));
% 
%                 if(Sensors(Sender(i)).E>0)
%                     rap=rap+1;                 
%                 end
%             
%            end 
%        end
  