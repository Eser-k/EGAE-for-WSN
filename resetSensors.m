function Sensors=resetSensors(Sensors,Model)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Resets each sensor’s cluster‐head state at the start of a new round:
 %    • MCH:    Set the “member of cluster head” field back to the sink ID,
 %              indicating no cluster head assigned yet.
 %    • type:   Mark every node as a normal sensor ('N').
 %    • dis2ch: Reset the stored distance to cluster head to Inf so that 
 %              the next CH election can overwrite it with the closest CH.

    n=Model.n;
    for i=1:n
        Sensors(i).MCH=n+1;
        Sensors(i).type='N';
        Sensors(i).dis2ch=inf;
    end
    
end