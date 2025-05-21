function Receiver=findReceiver(Sensors,Model,Sender,SenderRR)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Determines which sensors are within communication range of a 
 %  given sender.
 %
 %  This function computes the Euclidean distance from the specified 
 %  Sender node to every other sensor in the network. 
 %  It returns the IDs of all sensors whose distance from the 
 %  Sender is less than or equal to the Sender’s radio range.
 %  The Sender itself is excluded from the receiver list.

    Receiver=[];
    
    n=Model.n;
    D=zeros(1,n);
    
    for i=1:n
             
        D(i)=sqrt((Sensors(i).xd-Sensors(Sender).xd)^2+ ...
            (Sensors(i).yd-Sensors(Sender).yd)^2);                  
    end 
    
    for i=1:n
             
        if (D(i) <= SenderRR & Sender~=Sensors(i).id)
            Receiver=[Receiver,Sensors(i).id]; %#ok
        end
                      
    end 
    
end
