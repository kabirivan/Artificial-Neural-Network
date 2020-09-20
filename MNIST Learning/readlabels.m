function labels = readlabels(labelfile)

    flabels = fopen(labelfiles, 'r', 'b');
    header = fread(flabels, 1, 'int32');
    
    if header ~= 2049
        error('Invalid label file header');    
    end
    
    numlabels = fread(flabels, 1, 'int32');
    labels = fread(flabels, numlabels, 'uint8');
    
    fclose(flabels);


end

