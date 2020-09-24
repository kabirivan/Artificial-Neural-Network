%Cargar imagenes y etiquetas
testimages=readimages('t10k-images-idx3-ubyte');
testlabels=readlabels('t10k-labels-idx1-ubyte');

%Numero de imagenes para probar
%numtestimages=size(testimages,3);
numtestimages=10;
%1 es leakyrelu, 2 es sigmoide
fun=1;

%Numero de nodos de salida
testoutputnodes=10;

successtest=0;
   
    %Testingforward propagation 
    for i=1:numtestimages
        
        figure(i)
        img = imresize(testimages(:,:,i), [227 227]);
        imshow(img)
        
        ytest=zeros(testoutputnodes,1);
        ytest(testlabels(i)+1,1)=1;
        for row=1:size(testimages,1)
            for columns=1:size(testimages,2)
                atestinput(columns+(row-1)*size(testimages,2),1)=testimages(row,columns,i);
            end
        end
        
        %Forward Propagation
        %Capa de entrada
        ztest(:,1)=winput*atestinput+b(:,1);
        atesth(:,1)=activation(ztest(:,1),fun,0);
        %Capas escondidas
        if hlayers>1
            for l=2:hlayers
                ztest(:,l)=wh(:,:,l-1)*atesth(:,l-1)+b(:,l);
                atesth(:,l)=activation(ztest(:,l),fun,0);
            end
        end
        %Capa de salida
        ztestoutput=woutput*atesth(:,hlayers)+boutput;
        atestoutput=activation(ztestoutput,fun,0);
        
        [val, idx] = max(atestoutput);
        
        successtest=successtest+[find(atestoutput==max(atestoutput))==(testlabels(i)+1)];
        responses(i,:) = [testlabels(i), idx-1]
    end
   
    successtestrate = successtest/numtestimages
    
      