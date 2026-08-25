% Package orientation: Plotting/figure script. It consumes saved result structs or generated arrays and formats diagnostics or publication figures without changing model training.

% RECURRENT SNN DIAGRAM — tidy, E=I, dense recurrent
% Inputs/outputs indicated only by arrows into inputs and out of outputs

%% Figure setup
figure('Color','w');
hold on; axis equal; axis off;

% X positions
inX  = 0;                 % Input column
hidX = 3;                 % Hidden population centre
outX = hidX + 0.5*(9-hidX);   % Output 50% closer to hidden -> 6

% Input/Output y
yInOut   = [1, 0, -1.5];
inputPos  = [repmat(inX,3,1),  yInOut'];
outputPos = [repmat(outX,3,1), yInOut'];

% ----- Hidden population on a ring (E = I) -----
nHidden = 6;                               % compact & legible
theta = linspace(0,2*pi,nHidden+1)'; theta(end)=[];
a = 1.2; b = 2.2;                          % ellipse radii (x,y) for ring
hiddenPos = [hidX + a*cos(theta), b*sin(theta)];

% Translucent blue population hull fully enclosing the ring
wh = [2.1*(a+0.25), 2.1*(b+0.35)];
populationCloud([hidX,0], wh, [0 0.4470 0.7410], 0.10);

% Pale-green input box + amber output box
shadedBox(inX,  yInOut, [0.6 0.90 0.3]);
shadedBox(outX, yInOut, [0.9290 0.6940 0.1250]);

% Guide dots (light) for column hints
plot([inX inX inX],   linspace(-1.1,-0.5,3),'k.','markersize',12.5);
plot([outX outX outX],linspace(-1.1,-0.5,3),'k.','markersize',12.5);

% -------- Bias rail FROM BELOW (with upward dashed arrows) --------
yBias = -3.4;                 % place rail below population hull
railW = 3.2; railH = 0.6;
drawBiasRail(hidX, yBias, railW, railH, 'Bias');

for i = 1:size(hiddenPos,1)
    src = [hiddenPos(i,1), yBias + railH/2];
    dst = [hiddenPos(i,1), hiddenPos(i,2) - 0.30];
    drawDashedArrow(src, dst, [0.8 0.1 0.1], 1.5);
end

% ---- Forward connections ----
altHidden = 1:2:nHidden;                          
connectLayersToSubset(inputPos, hiddenPos, altHidden, 1.4);
connectRecurrentDense(hiddenPos, 0.55, [0.35 0.35 0.35], 0.5);   % fully dense within hidden
connectLayersFromSubset(hiddenPos, outputPos, altHidden, 1.4);

% ---- Neurons (balanced E/I) ----
nExc = nHidden/2;
hiddenTypes = [repmat("exc",nExc,1); repmat("inh",nHidden-nExc,1)];
hiddenTypes = hiddenTypes(randperm(nHidden));
drawNeurons(inputPos,  "black");
drawNeurons(hiddenPos, hiddenTypes);
drawNeurons(outputPos, "black");

% ---- I/O arrows only (no text labels) ----
% Arrows INTO input neurons (from the left)
ioInOffset  = 0.9;           % how far left to start
for i = 1:size(inputPos,1)
    a = [inputPos(i,1)-ioInOffset, inputPos(i,2)];
    b = [inputPos(i,1)-0.28,       inputPos(i,2)];   % stop near circle edge
    drawStraightArrow(a, b, [0 0 0], 1.5);
end

% Arrows OUT OF output neurons (to the right)
ioOutOffset = 1;           % how far right to end
for i = 1:size(outputPos,1)
    a = [outputPos(i,1)+0.28,        outputPos(i,2)];  % start near circle edge
    b = [outputPos(i,1)+ioOutOffset, outputPos(i,2)];
    drawStraightArrow(a, b, [0 0 0], 1.5);
end

%% Limits
xlim([-1.5, 8.2]); ylim([-6.0, 3.2]);

%% Out-of-plane feedback cues (n-th output -> n-th input), BENT arcs
fbCol = [0.49 0.18 0.56];   % purple to distinguish from main wiring
nFB   = min(size(outputPos,1), size(inputPos,1));

tail  = 0.75;               % horizontal reach of each dotted arc
bow   = 0.35;               % vertical bend magnitude

for k = 1:nFB
    % ----- OUTPUT side: bend upward (as if going INTO the page) and end with ⊗
    y  = outputPos(k,2);
    p0 = [outputPos(k,1)+ioOutOffset, y];   % where the solid output arrow ends
    p1 = p0 + [tail, 0.4];                    % arc end (further right)
    c  = (p0 + p1)/2 + [0, -1*bow];           % control point above the line
    drawDottedBezier(p0, c, p1, fbCol, 2);
    text(p1(1)+0.08, p1(2), '⊗', 'FontSize',16, 'Color',fbCol, ...
        'HorizontalAlignment','left','VerticalAlignment','middle');

    % ----- INPUT side: start with ⊙ then bend downward (as if emerging OUT of page)
    y  = inputPos(k,2);
    p1 = [inputPos(k,1)-ioInOffset, y];     % where the solid input arrow starts
    p0 = p1 - [tail, -0.4];                    % arc start (further left)
    c  = (p0 + p1)/2 + [0, -bow];           % control point below the line
    text(p0(1)-0.08, p0(2), '⊙', 'FontSize',16, 'Color',fbCol, ...
        'HorizontalAlignment','right','VerticalAlignment','middle');
    drawDottedBezier(p0, c, p1, fbCol, 2);
end


% ===== Figure Legend =====
% 1) Solid black arrow — Input/output
% 2) Black dot — Input/output neuron
% 3) Red dot — Excitatory spiking neuron
% 4) Blue dot — Inhibitory spiking neuron
% 5) Solid black line — Encoder/decoder weight
% 6) Grey line — Recurrent weight
% 7) Dotted red arrow — Bias
% 8) Dotted purple line — (Optional) feedback channel

figure('Color','w'); hold on; axis off; axis equal;

% Layout
x0 = 0; x1 = 1.4; xLabel = 2.1;      % icon start, icon end, label x
y0 = 8; dy = 1.2;                    % top y and vertical spacing
r  = 0.3;                            % dot radius

% Colours
colBlack  = [0 0 0];
colRedExc = [0.8500 0.3250 0.0980];
colBlueInh= [0 0.4470 0.7410];
colGrey   = [0.35 0.35 0.35];
colPurple = [0.49 0.18 0.56];
colBias   = [0.8 0.1 0.1];

% 1) Solid black arrow — Input/output
y = y0 - 0*dy;
legendArrow([x0 y],[x1 y],colBlack,1.6);
text(xLabel,y,'Input/output','FontSize',13,'VerticalAlignment','middle');

% 2) Black dot — Input/output neuron
y = y0 - 1*dy;
rectangle('Position',[x0-r, y-r, 2*r, 2*r],'Curvature',[1 1],...
    'FaceColor',colBlack,'EdgeColor','k','LineWidth',1.2);
text(xLabel,y,'Input/output neuron','FontSize',13,'VerticalAlignment','middle');

% 3) Red dot — Excitatory spiking neuron
y = y0 - 2*dy;
rectangle('Position',[x0-r, y-r, 2*r, 2*r],'Curvature',[1 1],...
    'FaceColor',colRedExc,'EdgeColor','k','LineWidth',1.2);
text(xLabel,y,'Excitatory spiking neuron','FontSize',13,'VerticalAlignment','middle');

% 4) Blue dot — Inhibitory spiking neuron
y = y0 - 3*dy;
rectangle('Position',[x0-r, y-r, 2*r, 2*r],'Curvature',[1 1],...
    'FaceColor',colBlueInh,'EdgeColor','k','LineWidth',1.2);
text(xLabel,y,'Inhibitory spiking neuron','FontSize',13,'VerticalAlignment','middle');

% 5) Solid black line — Encoder/decoder weight
y = y0 - 4*dy;
plot([x0 x1],[y y],'-','Color',colBlack,'LineWidth',5);
text(xLabel,y,'Encoder/decoder weight','FontSize',13,'VerticalAlignment','middle');

% 6) Grey line — Recurrent weight
y = y0 - 5*dy;
plot([x0 x1],[y y],'-','Color',colGrey,'LineWidth',5);
text(xLabel,y,'Recurrent weight','FontSize',13,'VerticalAlignment','middle');

% 7) Dotted red arrow — Bias  (CHANGED)
y = y0 - 6*dy;
legendArrowLS([x0 y],[x1 y],colBias,2,':');
text(xLabel,y,'Bias','FontSize',13,'VerticalAlignment','middle');

% 8) Dotted purple line — (Optional) feedback channel
y = y0 - 7*dy;
plot([x0 x1],[y y],':','Color',colPurple,'LineWidth',5);
text(xLabel,y,'(Optional) feedback channel','FontSize',13,'VerticalAlignment','middle');

xlim([x0-0.5, xLabel+4.5]); ylim([y0-8.5*dy, y0+dy]);

% ---- helpers ----
function legendArrow(a,b,col,lw)
    if nargin<4, lw=1.4; end
    plot([a(1) b(1)],[a(2) b(2)],'-','Color',col,'LineWidth',lw); hold on;
    v = b - a; nv = norm(v); if nv==0, return; end
    v = v/nv; ah = 0.18; side = [v(2) -v(1)]*0.45;
    L = b - ah*(v + side); R = b - ah*(v - side);
    fill([b(1) L(1) R(1)],[b(2) L(2) R(2)],col,'EdgeColor',col);
end

function legendArrowLS(a,b,col,lw,ls)
    if nargin<5, ls='--'; end
    if nargin<4, lw=1.4; end
    plot([a(1) b(1)],[a(2) b(2)],'LineStyle',ls,'Color',col,'LineWidth',lw); hold on;
    v = b - a; nv = norm(v); if nv==0, return; end
    v = v/nv; ah = 0.18; side = [v(2) -v(1)]*0.45;
    L = b - ah*(v + side); R = b - ah*(v - side);
    fill([b(1) L(1) R(1)],[b(2) L(2) R(2)],col,'EdgeColor',col);
end



%% ---------- Helper functions ---------- %%
function drawNeurons(pos, types)
    if nargin<2, types="exc"; end
    if isstring(types) && isscalar(types), types=repmat(types,size(pos,1),1); end
    r=0.2;
    for i=1:size(pos,1)
        switch types(i)
            case "inh",  face=[0.8500 0.3250 0.0980];
            case "black",face=[0 0 0];
            otherwise,   face=[0 0.4470 0.7410];
        end
        rectangle('Position',[pos(i,1)-r,pos(i,2)-r,2*r,2*r],...
            'Curvature',[1 1],'EdgeColor','k','FaceColor',face,'LineWidth',1.3);
    end
end

function connectLayersToSubset(p1,p2,idx2,lw)
    if nargin<4, lw=1; end
    idx2 = idx2(idx2>=1 & idx2<=size(p2,1));
    for i=1:size(p1,1)
        for j=idx2
            plot([p1(i,1),p2(j,1)],[p1(i,2),p2(j,2)],'k','LineWidth',lw);
        end
    end
end

function connectLayersFromSubset(p1,p2,idx1,lw)
    if nargin<4, lw=1; end
    idx1 = idx1(idx1>=1 & idx1<=size(p1,1));
    for i=idx1
        for j=1:size(p2,1)
            plot([p1(i,1),p2(j,1)],[p1(i,2),p2(j,2)],'k','LineWidth',lw);
        end
    end
end

function connectRecurrentDense(pos,curvMag,color,lw)
    % Fully dense recurrent connectivity within 'pos' (i ~= j)
    if nargin<2, curvMag=0.5; end
    if nargin<3, color=[0.35 0.35 0.35]; end
    if nargin<4, lw=0.8; end
    n=size(pos,1);
    for i=1:n
        for j=1:n
            if i~=j
                sgn = (-1)^(i+j); % alternate curvature sign to reduce overlap
                drawCurvedArrowLS(pos(i,:), pos(j,:), sgn*curvMag, color, lw, '-');
            end
        end
    end
end

function drawCurvedArrowLS(a,b,curv,color,lw,ls)
    % Quadratic Bézier with arrowhead; linestyle selectable
    mid=(a+b)/2; dir=b-a; n=[-dir(2),dir(1)];
    nn=norm(n); if nn==0, return; end
    n=n/nn; ctrl=mid+curv*n;
    t=linspace(0,1,60);
    x=(1-t).^2*a(1)+2*(1-t).*t*ctrl(1)+t.^2*b(1);
    y=(1-t).^2*a(2)+2*(1-t).*t*ctrl(2)+t.^2*b(2);
    plot(x,y,'Color',color,'LineWidth',lw,'LineStyle',ls);
    v=[x(end)-x(end-1), y(end)-y(end-1)];
    nv=norm(v); if nv==0, return; end
    v=v/nv; ah=0.16; side=[v(2) -v(1)]*0.45;
    L=[b(1),b(2)] - ah*(v + side);
    R=[b(1),b(2)] - ah*(v - side);
    fill([b(1) L(1) R(1)],[b(2) L(2) R(2)],color,'EdgeColor',color);
end

function shadedBox(x,yC,col)
    rectangle('Position',[x-0.5,min(yC)-0.4,1,max(yC)-min(yC)+0.8],...
        'FaceColor',col,'EdgeColor','none','FaceAlpha',0.15);
end

function populationCloud(center,wh,col,alphaVal)
    t = linspace(0,2*pi,200);
    x = center(1) + (wh(1)/2)*cos(t);
    y = center(2) + (wh(2)/2)*sin(t);
    patch(x,y,col,'EdgeColor','none','FaceAlpha',alphaVal);
end

function drawBiasRail(cx, y, w, h, labelStr)
    x = (cx - w/2);
    rectangle('Position',[x, y-h/2, w, h], 'Curvature',[0.2 0.7], ...
        'FaceColor',[0.25 0.25 0.25], 'EdgeColor','k', 'LineWidth',1.2);
    text(cx-0.1, y, ['  ',labelStr], 'FontSize',17, 'Interpreter','tex', ...
        'HorizontalAlignment','center','VerticalAlignment','middle','Color','w');
end

function drawDashedArrow(a,b,col,lw)
    if nargin<4, lw=1; end
    plot([a(1) b(1)],[a(2) b(2)],'--','Color',col,'LineWidth',lw);
    v = b - a; nv = norm(v); if nv==0, return; end
    v = v / nv;
    ah = 0.18; side = [v(2) -v(1)]*0.45;
    L = b - ah*(v + side); R = b - ah*(v - side);
    fill([b(1) L(1) R(1)],[b(2) L(2) R(2)],col,'EdgeColor',col);
end

function drawStraightArrow(a,b,col,lw)
    if nargin<4, lw=1.2; end
    plot([a(1) b(1)],[a(2) b(2)],'-','Color',col,'LineWidth',lw);
    v = b - a; nv = norm(v); if nv==0, return; end
    v = v / nv;
    ah = 0.16; side = [v(2) -v(1)]*0.45;
    L = b - ah*(v + side); R = b - ah*(v - side);
    fill([b(1) L(1) R(1)],[b(2) L(2) R(2)],col,'EdgeColor',col);
end
function drawDottedBezier(a,c,b,color,lw)
    % Quadratic Bézier dotted arc a->b with control point c
    if nargin<5, lw=1.2; end
    t = linspace(0,1,60);
    x = (1-t).^2*a(1) + 2*(1-t).*t*c(1) + t.^2*b(1);
    y = (1-t).^2*a(2) + 2*(1-t).*t*c(2) + t.^2*b(2);
    plot(x,y,':','Color',color,'LineWidth',lw);
end

