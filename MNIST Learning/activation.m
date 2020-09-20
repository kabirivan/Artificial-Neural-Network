function output = activation(input,fun,d)

% RELU fucntion
    if fun == 1
        output = input;
        
        if d == 0
            output(input<0) = 0.1 * input(input<0);
            
        elseif d==1
            output(input<0) = 0.1;
            output(input>=0) = 1;
            
        end
        
% SIGMOID function  
    elseif fun == 2
        output = 1./(1+exp(-1.*input));
        
        if d==1
            output = output.*(1.-output);
        end
        
    end





end