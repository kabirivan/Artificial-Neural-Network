clc
clear all

% Cargar imagenes
images = readimages('train-images-idx3-ubyte');
labels = readlabels('train-labels-idx1-ubyte');

% Definir rango de aprendizaje del backpropagation
learningrate = 0.001;

% Numero de imagenes del dataset MNIST
numImages = size(images,3);

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