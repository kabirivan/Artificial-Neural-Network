%loading the images and labels
testimages=readimages('t10k-images-idx3-ubyte');
testlabels=readlabels('t10k-labels-idx1-ubyte');

%Number of images from thhe training set we want to work on
%numtestimages=size(testimages,3);
numtestimages=100;
%1 is leakyrelu, 2 is sigmoid
fun=1;

%Number of nodes in input layer, outplayer and hidden layer
testoutputnodes=10;

successtest=0;
   
    %Testingforward propagation 
    for i=1:numtestimages
        ytest=zeros(testoutputnodes,1);
        ytest(testlabels(i)+1,1)=1;
        for row=1:size(testimages,1)
            for columns=1:size(testimages,2)
                atestinput(columns+(row-1)*size(testimages,2),1)=testimages(row,columns,i);
            end
        end
        
        %Forward Propagation
        %Input Layer
        ztest(:,1)=winput*atestinput+b(:,1);
        atesth(:,1)=activation(ztest(:,1),fun,0);
        %Hidden layers
        if hlayers>1
            for l=2:hlayers
                ztest(:,l)=wh(:,:,l-1)*atesth(:,l-1)+b(:,l);
                atesth(:,l)=activation(ztest(:,l),fun,0);
            end
        end
        %Output layer
        ztestoutput=woutput*atesth(:,hlayers)+boutput;
        atestoutput=activation(ztestoutput,fun,0);
        
        successtest=successtest+[find(atestoutput==max(atestoutput))==(testlabels(i)+1)];
        
    end
   
    successtestrate=successtest/numtestimages
    
      