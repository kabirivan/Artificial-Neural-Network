function imgs = readimages(imagefile)
    
    %Proceso para adquirir imagenes
    fimages = fopen(imagefile, 'r', 'b');
    header = fread(fimages, 1, 'int32');
    
    if header ~= 2051
        error('Invalid image file header');
    end
    
    numimages = fread(fimages, 1, 'int32');
    rows = fread(fimages, 1, 'int32');
    columns = fread(fimages, 1, 'int32');

    imgs = zeros([rows, columns, numimages]);
    
    
    for i = 1:numimages
        for y=1:rows
            imgs(y,:,i) = fread(fimages, columns, 'uint8');
            
        end
    end
    
    fclose(fimages);
end
       
    