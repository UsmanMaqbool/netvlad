function mat_boxes = leo_slen_increase_boxes(bbx,wd,hh)
    bboxes=[];
    gt=[111	98	25	101];

    b_size = size(bbx,1); 
    for ii=1:b_size
         bb=bbx(ii,:);
         square = bb(3)*bb(4);
         if square <2*gt(3)*gt(4)
            bboxes=[bbx;bb];
         end
    end
    mat_boxes = uint8(bboxes/16); 
        fprintf( '=> %i', length(mat_boxes));
        %size(mat_boxes) (if boxes are less then 50 -> create empty boxes
        while (size(mat_boxes) < 50)
            mat_boxes_add = [0 0 wd/16-1 hh/16-1 0]; 
            mat_boxes( end+1, : ) = mat_boxes_add; 
            %size(mat_boxes);
        end
        
end