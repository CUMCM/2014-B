% CreativeFoldingTable Matlab script for 2014 CUMCM problem B
%% ------------------------------------------------------------------------
% zhou lvwen: zhou.lv.wen@gmail.com
% september 13, 2014
%% ------------------------------------------------------------------------
% 2014 CUMCM - B: Creative Folding Table: CreativeFoldingTable.m
% 
% http://vimeo.com/groups/etsas/videos/54411397
% http://www.robertvanembricqs.com/#!rising-side-table/c1z4p
%% ------------------------------------------------------------------------
%
%   y                        |<------L1---->|            | d = 2.5cm
%   ^                                                    V
% R +    --------|-------=====--------|------- y1[1]    ---
%   |    --------|----===========-----|------- y1[2]    ---
%   |    --------|---=============----|-------  .        ^
% 0 +    --------|--===============---|-------  .        |
%   |    --------|---=============----|-------  .
%   |    --------|----===========:----|-------  .
%-R +    --------|-------=====:--:----|------- y1[n]
%   |                         :  :                        a = x0[n]   = x1
%   |                         :  :                        b = x0[n-1] = x2
%   +----+-----------------+--+--+-----------+-----> x    .
%       -W                  0  a  b          W            .
%
%% ------------------------------------------------------------------------
% USAGE: [beams,table] = CreativeFoldingTable( H, W, D, R, shape, name,...
%                                                     isanimate, isresults)
%                    H = height of the folding table
%                    W = width of the rectangular flat 
%                    D = thickness of the rectangular flat 
%                    R = parameters for desktop
%                shape = shape of the desktop: 'ellipse' or 'polygon'
%                 name = name of output files
%            isanimate = is write a gif animation file?: true or false
%            isresults = is write the results in to a dat file?
%% ------------------------------------------------------------------------
% shape:   'ellipse':   x^2/a^2 + y^2/b^2 = 1       =====>   R = [a, b]
%
%
% shape:   'polygon':     x3,y3)  _______ (x2,y2)   =====>   R = [x1, y1
%                                /       \                        x2, y2
%                       (x4,y4) /         \ (x1,y1)               x3, y3
%                               \         /                       x4, y4
%                        (x5,y5) \_______/ (x6,y6)                x5, y5
%                                                                 x6, y6
%                                                                 x1, y1]
%% ------------------------------------------------------------------------

function [beams,table] = CreativeFoldingTable( H, W, D, R, shape, name,...
                                               isanimate, isresults)
% H = height of the folding table; W = diameter of circular table top
if nargin == 0; 
    H  = 53; W = 50; D =  3; 
    shape = 'ellipse'; name = 'circle';
    R = [W/2  W/2];
end

if nargin < 7
    isanimate = true;  % is write a gif animation file?
    isresults = true;  % is write the results in to a dat file?
end

if isanimate; nsteps = 80; else; nsteps = 1; end;

global namefile; namefile = ['Folding-Table-',name];

table = desktop(shape, H, R);
shapefun = @(y,sign)tableshape(shape, R, y, sign);

[Lleft , w] = optimizedparameter(W, H, D, shapefun, -1);
[Lright, w] = optimizedparameter(W, H, D, shapefun,  1);
rect = rectflat(W,[Lleft,Lright],D);

w = w;                  % width of each wooden beam.      unit: cm
d = D;                  % thickness of the wooden beam.   unit: cm
nbeams = rect.w/w;      % half number of beams
beams = beam(nbeams, w, d, rect, shapefun);

% steal bars
bars = steelbar(beams, table, rect);

% show talbe, beams and stealbar 
table = plotable(rect, beams, table);
hbeams = plotbeams(beams);
hbar = plotsteelbar(bars, rect,[]);

% plot foot curve
hfoot = plotfootcurve(table,bars,rect,shapefun,[]);

if isanimate; frames = writeanimate([],0.1,'start'); end

for ti = 1:nsteps
    delete(hbeams) 
    for sign = [-1 1]
        j = (sign+1)/2 + 1;
        
        theta = bars(j).rotate/nsteps*ti*sign;
        % update position of the steal bar
        bars(j).x = bars(j).xrot + bars(j).r*cos(theta*sign+bars(j).dtheta)*sign;
        bars(j).z = bars(j).zrot + bars(j).r*sin(theta*sign+bars(j).dtheta);
        
        % determined positions of each beams.
        for i = 1:nbeams
            holenx = bars(j).x-beams(i,j).x0;
            holenz = bars(j).z-beams(i,j).z0;
            holen = sqrt( holenx.^2 + holenz.^2 -(beams(i,j).d/2)^2);
            
            beta = atan(holenz/holenx);
            if (holenz)<0 & (holenx*sign)<0; beta = -pi+beta;end
            
            dbeta = atan(beams(i,j).d/2/holen)*sign;
            
            beams(i,j).x1 = beams(i,j).x0 + beams(i,j).l*cos(beta-dbeta)*sign;
            beams(i,j).z1 = beams(i,j).z0 + beams(i,j).l*sin(beta-dbeta)*sign;
            beams(i,j).holen = holen;
        end
    end

    hbar = plotsteelbar(bars, rect, hbar);
    hbeams = plotbeams(beams);
    hfoot = plotfootcurve(table,bars,rect,shapefun,hfoot);
    drawnow
    if isanimate; frames = writeanimate(frames,0.1,'fold'); end
end

if isanimate;
    frames = writeanimate(frames,0.1,'rot');
    frames = writeanimate(frames,0.1,'end');
end

if isresults; writeresults(rect,table,bars,beams); end

% ------------------------------------------------------------------------

function [L, w] = optimizedparameter(W, H, D, shapefun, sign)
%                                             %
%             0              x0    R          %
%      +------+--------------+-----+-----> x  % subject to:
%      |                     :     :          %    2.5 <= w <= 5
%      |    table_top_face___:____ :          %    D/2 <= w <= 2*D
% 0    |_     _______________|\____|          %    n = W/w in [10,11,...30]
%      |                    \ \               %    theta < 75/180*pi
%      |                     \ \ theta        %    xend  > x0 +(R-x0)/2
%      |         longest beam \ \             %    xend  < R + D
%      |                       \ \            %     
% -H+D |_                       \_\           % min theta, L   
%      |                           xend       %    
%      V                                      %
%      z                                      %
%

R = W/2;                 % table radius
L = max(R,H):5*max(R,H); % range of rectangle's length

% determined the range of beams' width and number of beams
w = [max(2.5,D/2) min(D*2,5)];
n = round(W./w/2)*2;
n = max(n,10);
n = min(n,30);
n = n(2):2:n(1);
w = W./n;

% search domain
[L,w] = meshgrid(L,w);
L = L(:); w = w(:); n = W./w;

% the longest beam's start point
y0 = R-w./2;
x0 = abs(shapefun(y0,sign));

sint = (H-D)./(L-x0);
[sint, L, w, x0,n] = mask(abs(sint)<1, sint, L, w, x0, n);
theta = asin(sint); 

% x0 +(R-x0)/2 < xend  < R + D
xend = x0 + (L-x0).*cos(theta);
[theta, L, w, x0,n] = mask((xend<=R+D)&(xend>=(R+x0)/2), theta, L, w, x0,n);

% minimize theta & minimize L & theta < 75/180pi if possible
if any(theta<(75/180*pi))
    [theta, L, w, x0,n] = mask(theta<(75/180*pi), theta, L, w, x0,n);
    [theta, L, w, x0,n] = mask(L==min(L), theta, L, w, x0,n);
    [theta, L, w, x0,n] = mask(theta==min(theta), theta, L, w, x0,n);
else
    [theta, L, w, x0,n] = mask(theta==min(theta), theta, L, w, x0,n);
    [theta, L, w, x0,n] = mask(L==min(L), theta, L, w, x0,n);
end

% ------------------------------------------------------------------------

function varargout = mask(masks, varargin)
for i = 1:length(varargin);
    vari = varargin{i};
    varargout{i} = vari(masks);
end

% ------------------------------------------------------------------------

function x = tableshape(shape, R, y, sign)
%
% get x for give y and table shape. sign = -1 for left (x<0); sign = 1 for 
% right (x>0) 
%

if nargin == 3; sign = 1; end

switch shape
    case 'ellipse' % circle (Rx==Ry) and ellipse
        Rx = R(1); Ry = R(2);
        y = y./Ry;
        x = sqrt(1-y.^2)*Rx*sign;
    case 'polygon' % polygon: rectangles, hexagon, octagon...
        x1 = R(:,1); y1 = R(:,2);
        x2 = x1([2:end 1]);
        y2 = y1([2:end 1]);
        
        x = zeros(size(y));
        for i = 1:length(y)
            if sign>0
                j = find(xor(y(i)>=y1,y(i)>y2));
                j1 = find( (x1(j)+x2(j))>=0 );
                j = j(j1(1));
            else
                j = find(xor(y(i)>y1,y(i)>=y2));
                j1 = find( (x1(j)+x2(j))<=0 );
                j = j(j1(1));
            end

            if   abs(y2(j)-y1(j))<1e-10;
                if sign>0; x(i) = max(x1(j), x2(j));end
                if sign<0; x(i) = min(x1(j), x2(j));end
            else
                x(i) = x1(j) - (x2(j)-x1(j))/(y2(j)-y1(j))*(y1(j)-y(i));
            end

        end
    otherwise
        error('Unknown shape, shape should be "ellipse" or "polygon"');
end

% ------------------------------------------------------------------------

function obj = rectflat(W,L,D)
% rectangular flat
%
%    y
%    ^     ___________________________________
%  W_|    /__________________________________/| D
%    |    |           /  |  \                ||
%    |    |          /   |   \               ||
%    |    |          \   |   /               ||
%  0_|_   |___________\__|__/________________|/
%    |                   :
%    +----+--------------+-------------------+----->
%       -L(1)            0                  L(2)
%
obj.w =  W;            % width of the rectangular.       unit: cm
obj.l =  L;            % length of the rectangular.      unit: cm
obj.d =  D;            % thickness of the rectangular.   unit: cm

% ------------------------------------------------------------------------

function obj = beam(n, w, d, rect, shapefun)

R = rect.w/2;
L = rect.l;

for sign = [-1,1]
    j = (sign+1)/2 + 1;
    % inti postion of steel bar
    y0 = -R+w/2;   x0 = shapefun(y0,sign);
    xbar = (x0 + L(j)*sign)/2;

    for i = 1:n
        obj(i,j).id = i + (j-1)*n;     % ID
        obj(i,j).w  = w;               % width
        obj(i,j).d  = d;               % thickness
        
        y0 = -R+w/2 + w*(i-1);         % origin point
        x0 = shapefun(y0,sign);
        obj(i,j).y0 = y0;
        obj(i,j).x0 = x0;
        obj(i,j).z0 = 0;
        
        obj(i,j).x1 = sign*L(j);       % end point
        obj(i,j).y1 = y0;
        obj(i,j).z1 = 0;
        
        obj(i,j).l  = abs(obj(i,j).x1 - obj(i,j).x0);  % length of the beam
        
        obj(i,j).holep = abs(xbar-x0); % relative position of slot hole's origin
        obj(i,j).holen = abs(xbar-x0); % relative position of slot hole's end
    end
end

% ------------------------------------------------------------------------

function h = plotbeams(beams)
% plot each beams and lengths of the slot holes on the wooden beams

h = [];
for sign = [-1 1]
    j = (sign+1)/2 + 1;
    x0 = cat(1,beams(:,j).x0); x1 = cat(1,beams(:,j).x1);
    y0 = cat(1,beams(:,j).y0); y1 = cat(1,beams(:,j).y1);
    z0 = cat(1,beams(:,j).z0); z1 = cat(1,beams(:,j).z1);
    w  = cat(1,beams(:,j).w)/1.05;  d  = cat(1,beams(:,j).d);
    
    theta = -atan( (z0-z1)./(x0-x1) )*sign;
    mask = (z1-z0)<0 & ((x1-x0)*sign)<0;
    theta(mask) = -pi + theta(mask);
    
    dx = d.*sin(theta)*sign;
    dz = d.*cos(theta);
    
    x01 = x0; x02 = x0 + dx;
    x11 = x1; x12 = x1 + dx;
    z01 = z0; z02 = z0 + dz;
    z11 = z1; z12 = z1 + dz;
    y01 = y0 - w/2; y02 = y0 + w/2;
    y11 = y1 - w/2; y12 = y1 + w/2;
    
    x = [x01 x01 x02 x02; x11 x11 x12 x12; x01 x01 x11 x11;
         x02 x02 x12 x12; x01 x02 x12 x11; x01 x02 x12 x11];
    y = [y01 y02 y02 y01; y11 y12 y12 y11; y01 y02 y12 y11;
         y01 y02 y12 y11; y01 y01 y01 y01; y12 y12 y12 y12];
    z = [z01 z01 z02 z02; z11 z11 z12 z12; z01 z01 z11 z11;
         z02 z02 z12 z12; z01 z02 z12 z11; z01 z02 z12 z11];
    
    h1 = fill3(x',y',z','red');
    
    % slot holes on the wooden beams
    holep = cat(1,beams(:,j).holep);
    
    scale = holep./sqrt((x0-x1).^2 + (z0-z1).^2);
    xh1 = x0 + (x1-x0).*scale;
    yh1 = y0 + (y1-y0).*scale;
    zh1 = z0 + (z1-z0).*scale;
    
    len = cat(1,beams(:,j).holen);
    scale = len./sqrt((x0-x1).^2 + (z0-z1).^2);
    
    xh2 = x0 + (x1-x0).*scale;
    yh2 = y0 + (y1-y0).*scale;
    zh2 = z0 + (z1-z0).*scale;
    
    h2 = plot3([xh1+dx xh2+dx]',[yh1 yh2]',[zh1+dz,zh2+dz]','-w','linewidth',2);
    
    h = [h; h1; h2];
end

% ------------------------------------------------------------------------

function obj = desktop(shape, H, R)
obj.shape = shape;
obj.c = [0,0,0];           % center of the bottom face of the table.
obj.h = H;                 % height of the folding table.    unit: cm
obj.r = R;                 % radius of the table.            unit: cm

% ------------------------------------------------------------------------

function table = plotable(rect, beams, table)
% plot table face

x0 = cat(1,beams.x0);
x0(end/2+1:end) = flipud(x0(end/2+1:end));
y0 = cat(1,beams.y0);
y0(end/2+1:end) =  flipud(y0(end/2+1:end));
z0 = cat(1,beams.z0);
w0 = cat(1,beams.w)/2;
w0(end/2+1:end) = -flipud(w0(end/2+1:end));
h0 = rect.d;

x = [x0 x0]'; y = [y0-w0 y0+w0]'; z = [z0 z0]';

table.x = [x(:) x(:)];
table.y = [y(:) y(:)];
table.z = [z(:) z(:)+h0];

figure('color',[1,1,1]); box on;
fill3(table.x,table.y,table.z,'y');
hold on; box on;
% draw the edge face of the table
x = [x0 x0 x0 x0 x0];
y = [y0-w0 y0+w0 y0+w0 y0-w0 y0-w0];
z = [z0 z0 z0+h0 z0+h0 z0];

l0 = (-x0+x0([end 1:end-1]));

x = [x; x0 x0+l0 x0+l0 x0 x0];
y = [y; y0-w0 y0-w0 y0-w0 y0-w0 y0-w0];
z = [z; z0 z0 z0+h0 z0+h0 z0];

fill3(x',y',z','y');
axis image; view(40,30)
axis([-rect.l(1)-5 rect.l(2)+5 -rect.w/2-5 rect.w/2+5 -table.h 10])
light('position',[0,40,60]); lighting phong; 

% ------------------------------------------------------------------------

function obj = steelbar(beams, table, rect)

for sign = [-1 1]
    j = (sign+1)/2 + 1;
    beamj = beams(1,j);
    % The reinforcing steel bar of each group is fixed in the middle of two
    % outermost wooden legs. this means the rotation radius of the two outermost
    % legs should be:
    d = beamj.d/2;
    obj(j).r = sqrt( (beamj.l/2)^2 + d^2 );
    % The height of the folding table is 53 cm. this means the two outermost
    % legs should rotate by the fellowing angle:
    obj(j).rotate = -asin((table.h-rect.d)/beamj.l);
    
    % rotation axis of the steal bar
    obj(j).xrot = beamj.x0;
    obj(j).zrot = beamj.z0;
    
    obj(j).dtheta = asin(d/obj(j).r);
    
    % position of the steal bar
    obj(j).x = obj(j).xrot + obj(j).r*cos(obj(j).dtheta)*sign;
    obj(j).z = obj(j).zrot + obj(j).r*sin(obj(j).dtheta);
end

% ------------------------------------------------------------------------

function h = plotsteelbar(bars, rect, h)
% plot stee bar 

xbar = cat(1,bars.x); xbar = [xbar xbar]';
zbar = cat(1,bars.z); zbar = [zbar zbar]';

if ishandle(h)
    set(h(1),'xdata', xbar(:,1),'zdata',zbar(:,2));
    set(h(2),'xdata', xbar(:,2),'zdata',zbar(:,2));
else
    ybar = [-0.53 0.53 ]'*rect.w; ybar = [ybar ybar];
    h = plot3(xbar, ybar, zbar,'.b-','linewidth',5,'markersize',10);
end


% ------------------------------------------------------------------------

function h = plotfootcurve(table,bars,rect, shapefun, h)
% plot foot curve use parameter euqation. foot curve parameter equation:
%
% theta= -pi/2...pi/2
%
% x0 = R*cos(theta); y0 = R*sin(theta); z0 = 0*theta;
% dx = xm - x0; dz = zm - z0; dr = sqrt(dx^2 + dz^2 - (h/2)^2);
% beta = atan(dz/dx); dbeta = atan(h/2/dr);
% 
% xi = x0 + dx*cos(beta-dbeta)* (L/2 - x0)/dr;
% yi = y0
% zi = z0 + dz*cos(beta-dbeta)* (L/2 - x0)/dr;
%

d = rect.d/2;

for sign = [-1 1];
    j = (sign+1)/2 + 1;
    Ry = max(table.r(:,2)*sign);
    
    y0 = -Ry:Ry/25:Ry;
    x0 = shapefun(y0,sign);
    z0 = 0*y0;

    xm =  bars(j).x;
    zm =  bars(j).z;
    beamlen = rect.l(j) - abs(x0);
    
    dz = zm-z0;
    dx = xm-x0;

    dr = sqrt(dz.^2 + dx.^2-d.^2);
    beta = atan(dz./dx);
    
    beta(dz<0 & (dx*sign)<0) = beta(dz<0 & (dx*sign)<0)-pi;
    
    dbeta = atan(d./dr)*sign;

    xi = x0 + beamlen.*cos(beta-dbeta)*sign;
    yi = y0;
    zi = z0 + beamlen.*sin(beta-dbeta)*sign;
    if ishandle(h)&(length(h)>=j)
        set(h(j),'xdata',xi,'ydata',yi,'zdata',zi);
    else
        h(j) = plot3(xi,yi,zi,'-m','linewidth',2);
    end
end

% ------------------------------------------------------------------------

function frames = writeanimate(frames,dt,mode)
global namefile;
n = length(frames);
switch mode
    case 'start'
        frames = getframe(gcf);
    case 'fold'
        n = n + 1;  frames(n)=getframe(gcf);
    case 'rot'
        for i = 40:5:400
            view(i,30);
            n = n + 1; frames(n)=getframe(gcf);
        end
    case 'end'
        for i=1:n
            [image,map]=frame2im(frames(i));
            [im,map2]=rgb2ind(image,128);
            if i==1
                imwrite(im,map2,[namefile,'.gif'],'gif','writeMode',...
                       'overwrite','delaytime',dt,'loopcount',inf);
            else
                imwrite(im,map2,[namefile,'.gif'],'writeMode',...
                       'append','delaytime',dt);
            end
        end
end

% ------------------------------------------------------------------------

function writeresults(rect,table,bars,beams)
sketch = [...
'%%%% 2014 CUMCM-B: Creative Folding Table: parameters of beams   \n', ...
'%%                                                               \n', ... 
'%%     |<---x--->|<------l------>|                    unit: cm   \n', ...
'%%      ______________________________________     _ _           \n', ...
'%%     /____________________________________ /|   _/_  W         \n', ...
'%%     |         ________________           | |    |             \n', ...  
'%%     |top      |___slot hole___|    bottom| |    |  H          \n', ...  
'%%     |____________________________________|/    _|_            \n', ... 
'%%                                                               \n', ...
'%%     |<---------------- L --------------->|                    \n', ...
'%%                                                               \n'];
global namefile;

domain = [-rect.l(1)-5 rect.l(2)+5 -rect.w/2-10 rect.w/2+20];
figure('position',[100,100,800,500])
ax(1) = axes('Position',[0.075,0.1,0.85,0.8]);
htab = fill(table.x(:,1),table.y(:,1),'y');
axis image; axis(domain); box off
ax(2) = axes('Position',[0.075,0.1,0.85,0.8],'XAxisLocation','top', ...
                                 'YAxisLocation', 'right','Color','none');
axis image; axis([domain]); box off

x0 = cat(1,beams.x0);         y0 = cat(1,beams.y0);
w  = cat(1,beams.w);
x0bar = cat(1,beams.holep).*sign(x0) + x0; y0bar = y0;
x1bar = cat(1,beams.holen).*sign(x0) + x0; y1bar = y0;

x1 = zeros(size(x0)); x1(1:end/2) = -rect.l(1); x1(end/2+1:end) = rect.l(2);
y1 = y0;

hold on
hbeam = fill( [x0 x0 x1 x1]', [y0-w/2 y0+w/2 y0+w/2 y0-w/2]','r');

hbar = fill( [x0bar x0bar x1bar x1bar]',[y0-w/5 y0+w/5 y0+w/5 y0-w/5]','w');

legend([htab hbeam(1) hbar(1)],'table','beam','slot hole')

[xtick,index] = unique([table.x(:,1); x1bar]);
xtick = round(xtick*100)/100;
ytick = [table.y(:,1); y0bar]; ytick = ytick(index);

for i = [1:4:length(xtick)]
   plot([xtick(i)   xtick(i)],  [-abs(ytick(i)),  -rect.w],'--'); 
   plot([xtick(i+2) xtick(i+2)],[ abs(ytick(i+2)), rect.w],'--'); 
end

orientation = {'right','left'};
for i = 1:2
    xticki = sort([-rect.l(1); rect.l(2); 0; ...
                   xtick([(2*i-1):4:ceil(end/2) (ceil(end/2)+2*i-1):4:end])]);
    set(ax(i),'xtick',xticki,'xticklabel',{num2str(xticki)});
    
    xtickstr=get(ax(i),'XTickLabel');
    set(ax(i),'XTickLabel',[]);
    x =get(ax(i),'XTick'); y =get(ax(i),'YTick'); 
    y = repmat((i==1)*domain(3) + (i==2)*domain(4)+(i-1.5),length(x),1);
    text(x,y,xtickstr,'HorizontalAlignment',orientation{i},'rotation',90);
end
saveas(gcf, namefile, 'pdf');

fid = fopen([namefile,'.txt'], 'wt');
fprintf(fid,sketch);
fprintf(fid,'%%%% Number of beams: %i\n\n', length(beams)*2);
formatstring = ['%% %2s\t %6s\t %6s\t %6s\t %6s\t %6s\n'];

fprintf(fid,'%%%% size of rectangular flat: %4.2f X %4.2f X %4.2f \n\n', ...
                                           sum(rect.l), rect.w, rect.d);

fprintf(fid,formatstring, 'Id','L','W','H','x','l');

formatstring = ' %2i\t %6.2f\t %6.2f\t %6.2f\t %6.2f\t %6.2f\n';

Id = cat(1,beams.id);
L = cat(1,beams.l);
W = cat(1,beams.w);
D = cat(1,beams.d);
x = cat(1,beams.holep);
l = cat(1,beams.holen) - x;
data = [Id L W D x l]; 
fprintf(fid,formatstring, data');
fclose(fid);
