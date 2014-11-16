% 2014 CUMCM - B: Creative Folding Table: main.m
% ------------------------------------------------------------------------
% zhou lvwen: zhou.lv.wen@gmail.com
% september 13, 2014
% ------------------------------------------------------------------------

type = 'hexagon'; % 'circle', 'rectangle', hexagon', 'octagon', 'heart'

switch type
    case 'circle'
        H  = 53; W = 50; D =  3;
        R = [W/2 W/2];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'ellipse',type);
    case 'circle02'
        H  = 70; W = 80; D =  3;
        R = [W/2 W/2];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'ellipse',type);
    case 'ellipse'
        H  = 53; W = 50; D =  3;
        R = [W/1.75 W/2];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'ellipse',type);
    case 'ellipse02'
        H  = 53; W = 50; D =  3;
        R = [W/2.25 W/2];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'ellipse',type);
    case 'rectangle'
        H  = 53; W = 50; D =  3;
        theta = 0:pi/2:2*pi;
        x = W/2*cos(theta);
        y = W/2*sin(theta);
        R = [x' y'];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'polygon',type);
    case 'hexagon'
        H  = 53; W = 50; D =  3;
        shape = 'polygon';
        theta=0:pi/3:2*pi;
        x = W/2*cos(theta);
        y = W/2*sin(theta);
        x = x/max(x)*W/2;
        y = y/max(y)*W/2;
        R = [x' y'];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'polygon',type);
    case 'octagon'
        H  = 53; W = 50; D =  3;
        shape = 'polygon';
        theta=0:pi/4:2*pi;
        x = W/2*cos(theta);
        y = W/2*sin(theta);
        x = x/max(x)*W/2;
        y = y/max(y)*W/2;
        R = [x' y'];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'polygon',type);
    case 'heart'
        H  = 63; W = 50; D =  3;
        t = 0:pi/10:2*pi;
        y = 16*sin(t).^3;
        x = 13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t);
        y = y./max(y)*W/2;
        x = x./max(x)*W/2.75-x(y==max(y));
        xshift = x(y==max(y));
        x = x - xshift(1);
        R = [x' y'];
        [beams,table] = CreativeFoldingTable(H,W,D,R,'polygon',type);
end