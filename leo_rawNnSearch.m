function [ids, ds]= leo_rawNnSearch(q, db, k,db_names,queryid)
    % wrapper as yael_nn insists to return two values while I sometimes need just one (for anonymous functions..)
    if nargin<5, k=1; end
    k= min(k, size(db,2));
    
%    Query_name = db.qImageFns(queryid);
    %Top_ids_name = zeros(100,1);
    %for i=1:100
    %Top_ids_name(i,1) = db.dbImageFns(ids(i,1));
    %end
    [ids, ds]= leo_yael_nn(db, q, k, db_names,queryid);
end
