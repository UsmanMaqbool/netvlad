function res_im = net_box(i,feat_im,res_im,bboxes)

y1 = bboxes(1);
y2 = bboxes(1)+bboxes(3);
x1 = bboxes(2);
x2 = bboxes(2)+bboxes(4);
[ww,hh,cc] = size(gather(feat_im));
if x1 == 0
    x1 = 1;
end
if y1 == 0
    y1 = 1;
end
if x2 > ww
    x2 = ww;
end
if y2 > hh
    y2 = hh;
end
if bboxes(3) < 2 && x2 > ww
    x1 = x1-2;
end
if bboxes(4) < 2 && y2 > hh
    y1 = y1-2;
end
%fprintf('%i %i %i %i - %i %i \n',x1, x2,y1, y2, x2-x1,y2-y1);

res_im(i).x = feat_im(x1:x2,y1:y2,:);

end