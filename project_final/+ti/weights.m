function w = weights(recordingIDs)
[~,~,g]=unique(string(recordingIDs)); counts=accumarray(g,1);
w=1./counts(g); w=w/mean(w);
end
