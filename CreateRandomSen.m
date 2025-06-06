function CreateRandomSen(Model,Area)
%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

%% Modified by Eser Kayali
 % eser-k@web.de

 %  Generates a random deployment of sensor nodes within a rectangular area
 %  and saves their coordinates to 'Locations.mat'.

    n=Model.n;
    x=Area.x;
    y=Area.y;
    X=zeros(1,n);
    Y=X;
    for i=1:1:n
        X(i)=rand()*x;
        Y(i)=rand()*y;
    end
    save ('Locations','X','Y');

end