function play_segment(segmentID)
cfg=ti.config(); m=ti.manifest(cfg); row=m(m.segment_id==string(segmentID),:);
assert(height(row)==1,'Unknown segment ID.'); [y,fs]=ti.audio(cfg,row); sound(y,fs);
end
