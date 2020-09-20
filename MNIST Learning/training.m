clc
clear all

% Cargar imagenes
images = readimages('train-images-idx3-ubyte');
labels = readlabels('train-labels-idx1-ubyte');

% Definir rango de aprendizaje del backpropagation
learningrate = 0.001;

% Numero de imagenes del dataset MNIST
%numImages = size(images,3);
numImages = 100;

% 1 es leakyRELU, 2es sigmoide
fun=1;

% Numero de neuronas en la capa de entrada, salida y escodida
inputnodes = size(images,1)*size(images,2);
outputnodes = 10;
nodeshlayers=20;

% Numero de capas escondidas
hlayers = 2;

% Pesos iniciales de la capa de entrada
winput = normrnd(0,sqrt(2/inputnodes),nodeshlayers,inputnodes);

% Pesos iniciales de la capa escondida
wh = normrnd(0,sqrt(2/nodeshlayers),nodeshlayers,nodeshlayers,hlayers-1);

% Pesos iniciales de la capa de salida
woutput = normrnd(0,sqrt(2/nodeshlayers),outputnodes,nodeshlayers);

% Inicializacion de las biases
b = zeros(nodeshlayers,hlayers);
boutput = zeros(outputnodes,1);




j = 0
success = 0;


% Lazo que entrena la red neuronal hasta obtener un resultado satisfactorio

while abs(success)< 0.99
    j = j + 1;
    % Inicializacion del grandiente y las neuronas de la capas escondidas
    deltab = zeros(nodeshlayers,hlayers);
    deltaboutput = zeros(outputnodes,1);
    deltawoutput = zeros(outputnodes,nodeshlayers);
    deltawh = zeros(nodeshlayers,nodeshlayers,hlayers-1);
    deltawinput = zeros(nodeshlayers,inputnodes);
    
    ah = zeros(nodeshlayers,hlayers);
    error = 0;
    
    for i = 1:numImages
        y = zeros(outputnodes,1);
        y(labels(i)+1,1) = 1;
        
        for row = 1:size(images,1)
            for columns = 1:size(images,2)
                
                ainput(columns+(row-1)*size(images,2),1) = images(row,columns,i);
            end
            
        end
        
        % Forward propagation
        % Capa de entrada
        
        z(:,1)=winput*ainput+b(:,1);
        ah(:,1)=activation(z(:,1),fun,0);
        
        % Capa escondida
        if hlayers>1
            for l=2:hlayers
                z(:,l)=wh(:,:,l-1)*ah(:,l-1)+b(:,l);
                ah(:,l)=activation(z(:,l),fun,0);
            end
        end
        %Output layer
        zoutput=woutput*ah(:,hlayers)+boutput;
        aoutput=activation(zoutput,fun,0);
        
        %Backpropagation
        %Compute the gradient of the output layer
        deltaoutput=activation(zoutput,fun,1).*(aoutput-y);
        %Last hidden layer
        deltah(:,hlayers)=activation(z(:,hlayers),fun,1).*(woutput.')*deltaoutput;
        %If more than one hidden layer, backpropagate on these
        if hlayers>1
            for back=(hlayers):-1:2
                deltah(:,back-1)=activation(z(:,back-1),fun,1).*(wh(:,:,back-1).')*deltah(:,back);
            end
        end
        
        %Correct weights and biases matrices using the gradient of the cost
        %function or deltas
        deltawoutput=deltawoutput+deltaoutput*(ah(:,hlayers).');
        deltaboutput=deltaboutput+deltaoutput;
        
        %hidden layers
        if hlayers>1
            for prop=2:hlayers
                deltawh(:,:,prop-1)=deltawh(:,:,prop-1)+deltah(:,prop)*((ah(:,prop-1)).');
                deltab(:,prop)=deltab(:,prop)+deltah(:,prop);
            end
        end
        %first hidden layer
        deltab(:,1)=deltab(:,1)+deltah(:,1);
        deltawinput=deltawinput+deltah(:,1)*ainput.';
        
        error=error+(aoutput-y)'*(aoutput-y);
        success=success+[find(aoutput==max(aoutput))==(labels(i)+1)];
        
    end
    
    deltawoutput=1/numImages.*deltawoutput;
    deltawh=1/numImages.*deltawh;
    deltawinput=1/numImages.*deltawinput;
    deltab=1/numImages.*deltab;
    deltaboutput=1/numImages.*deltaboutput;
    success=success/numImages;
    
    winput=winput-learningrate.*deltawinput;
    woutput=woutput-learningrate.*deltawoutput;
    wh=wh-learningrate.*deltawh;
    b=b-learningrate.*deltab;
    boutput=boutput-learningrate.*deltaboutput;
    
    error
    success
    err(j)=error;
    
    %Adaptive learning rate
    if j>2
        if err(j)<err(j-1)
            learningrate=learningrate*1.01;
        else
            learningrate=learningrate*0.5;
        end
    end
end
    



