function Sender=findSender(Sensors,Model,Receiver)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272 

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Identifies all sensor nodes that send data to a given receiver.
 %
 %  This function scans through the list of sensors and returns the IDs of
 %  those whose currently assigned cluster head (MCH) matches the specified
 %  Receiver. It excludes the receiver itself from the sender list. 
 
    Sender=[];
 
    n=Model.n;
 
    for i=1:n

        if (Sensors(i).MCH==Receiver & Sensors(i).id~=Receiver)
            Sender=[Sender,Sensors(i).id]; %#ok
        end

    end 

end