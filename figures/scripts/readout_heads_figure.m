%% Readout Head SE-adLIF Schematic (publication-ready, no LaTeX)
% Draws the readout head used in the training loop:
%  - Network synapse (gamma_sr, gamma_sd) -> r -> W_out, e-prop on B
%
% Exports PDF/SVG; falls back to tempdir and PNG if needed.
% Set output_basename to '' to skip export.
% Package orientation: Plotting/figure script. It consumes saved result structs or generated arrays and formats diagnostics or publication figures without changing model training.

clear;
output_basename = 'ReadoutHead_SNN';   % e.g., 'ReadoutHead_SNN' or '' (empty) to skip export

%% Aesthetics (plain text only)
set(0,'DefaultTextInterpreter','none');
set(0,'DefaultLegendInterpreter','none');
set(0,'DefaultAxesFontName','Helvetica');
set(0,'DefaultTextFontName','Helvetica');
set(0,'DefaultTextFontWeight','bold');   % make all text bold by default
set(0,'DefaultAxesFontWeight','bold');

col.core   = [0.20 0.20 0.20];  % dark grey (core/notes)
col.net    = [0.00 0.45 0.74];  % blue (readout head)
col.accent = [0.30 0.30 0.30];  % neutral
lw.box     = 1.2;
lw.arrow   = 3;

%% Canvas
figure('Color','w','Units','pixels','Position',[100 100 1100 620]);
axes('Position',[0 0 1 1],'Visible','off');

%% Layout (normalised coordinates)
% X anchors
x.in    = 0.06;
x.enc   = 0.16;
x.core  = 0.32;
x.synN  = 0.52;   % synapse/r
x.headN = 0.68;   % readout head
x.lossN = 0.86;   % output/loss metrics

% Y anchors (moved entire figure down)
y.top = 0.62;   % readout row centre
y.mid = 0.42;   % Core centre

% Sizes
w.box = 0.10; h.row = 0.12; h.core = 0.22;

%% Blocks
% Input / teacher forcing
draw_box(x.in,  y.mid-0.06, 0.08, 0.12, [0.95 0.95 0.95], 'Input', lw.box);

% Encoders W_in
draw_box(x.enc+0.01, y.mid-0.06, 0.10, 0.12, [0.95 0.95 0.95], 'Encoder', lw.box);

% Spiking core (SE-adLIF + low-rank recurrence)
core_text = sprintf(['ALIF neuron dynamics & spike time interpolation']);
draw_box(x.core, y.mid-0.11, 0.16, h.core, [0.95 0.95 0.95], core_text, lw.box);

% Synapse
draw_box(x.synN, y.top-0.06, w.box, h.row, [0.88 0.95 1.00], ...
    'Synaptic filter', lw.box);

% Head (decoder)
draw_box(x.headN, y.top-0.06, w.box, h.row, [0.88 0.95 1.00], ...
    'Decoder', lw.box);

% Loss/Selection panel
draw_box(x.lossN, y.top-0.06, 0.11, h.row, [0.88 0.95 1.00], ...
   'Output', lw.box);

% NEW: Bias box
x.bias = x.core+0.02; y.bias = y.mid + 0.2; w.bias = 0.10; h.bias = 0.08;
draw_box(x.bias, y.bias, w.bias, h.bias, [0.95 0.95 0.95], 'Biases (B)', lw.box);

% NEW: Error signal box
x.err = 0.62; y.err = y.top+0.18; w.err = 0.09; h.err = 0.08;
draw_box(x.err, y.err, w.err, h.err, [0.96 0.96 1.00], 'Error signal', lw.box);

% NEW: Eligibility trace box
x.elig = x.core-0.17; y.elig = y.top + 0.18; w.elig = 0.12; h.elig = 0.08;
draw_box(x.elig, y.elig, w.elig, h.elig, [0.96 1.00 0.96], 'Eligibility trace', lw.box);

% NEW: Bias update box
x.bupd = x.bias; y.bupd = y.top + 0.17; w.bupd = 0.1; h.bupd = 0.09;
draw_box(x.bupd, y.bupd, w.bupd, h.bupd, [1.00 0.98 0.90], 'Bias update (e-prop)', lw.box);

%% Connecting arrows
% Input -> Encoders
draw_arrow(x.in+0.08, y.mid, x.enc+0.01,      y.mid, col.accent, lw.arrow, '-');
% Encoders -> Core
draw_arrow(x.enc+0.11, y.mid, x.core,    y.mid, col.accent, lw.arrow, '-');

% Spike stream label
text(x.core+0.13, y.mid+0.15, 'Spikes', 'FontSize',11,'Color',col.core,'FontWeight','bold');

% Core -> synapse
draw_arrow(x.core+0.16, y.mid+0.10, x.synN, y.top, col.net,  lw.arrow, '-');

% Synapse -> head
draw_arrow(x.synN+w.box, y.top, x.headN, y.top, col.net,  lw.arrow, '-');

% Head -> Loss
draw_arrow(x.headN+w.box, y.top, x.lossN, y.top, col.net,  lw.arrow, '-');

% NEW: Error signal arrow
draw_arrow(x.lossN, y.top+0.06, x.err+0.09, y.top+0.22, col.net, lw.arrow, '-');

% NEW: Bias arrow to ALIF
draw_arrow(x.bias+0.05, y.bias, x.bias+0.05, y.mid+0.11, col.core, lw.arrow, '-');

% NEW: Eligibility arrow
draw_arrow(x.core, y.mid +0.11, x.elig+0.05, y.elig, col.core, lw.arrow, '-');

% NEW: Bias update arrows
draw_arrow(x.err, y.err + h.err/2, x.bupd + 0.1, y.bupd + 0.05, col.net, lw.arrow, '-');
draw_arrow(x.elig+0.12, y.elig + h.elig/2, x.elig+0.19, y.elig + h.elig/2, col.core, lw.arrow, '-');
draw_arrow(x.bupd+0.05, y.bupd, x.bupd+0.05, y.bias + 0.08, col.core, lw.arrow, '-');

%% Feedback arrow
y_fb    = 0.96;
draw_arrow(x.headN + w.box + 0.14, y.top + 0.06, x.headN + w.box + 0.14, y_fb, col.net, lw.arrow, '--');
draw_arrow(x.headN + w.box + 0.14, y_fb, x.in + 0.04, y_fb, col.net, lw.arrow, '--');
draw_arrow(x.in + 0.04, y_fb, x.in + 0.04, y.mid + 0.06, col.net, lw.arrow, '--');
text(x.in + 0.35, y_fb + 0.02, 'Closed-loop feedback', 'FontSize',14, 'Color', col.net,'FontWeight','bold');

axis([0 1 0 1]); axis off; drawnow;

%% ===== Local helper functions =====
function draw_box(x,y,w,h,facecol,str,linewidth)
    annotation('rectangle','Units','normalized','Position',[x y w h], ...
               'FaceColor',facecol,'LineWidth',linewidth,'Color',[0.2 0.2 0.2]);
    annotation('textbox','Units','normalized','Position',[x y w h], ...
               'String',str,'FontSize',11,'FontWeight','bold','EdgeColor','none','Color',[0.1 0.1 0.1], ...
               'Interpreter','none','HorizontalAlignment','center','VerticalAlignment','middle');
end

function draw_arrow(x1,y1,x2,y2,color,linewidth,linestyle)
    annotation('arrow','Units','normalized','X',[x1 x2],'Y',[y1 y2], ...
               'Color',color,'LineWidth',linewidth,'LineStyle',linestyle);
end

function draw_patch(x,y,w,h,color)
    annotation('rectangle','Units','normalized','Position',[x y w h], ...
               'FaceColor',color,'EdgeColor',[0.2 0.2 0.2],'LineWidth',0.8);
end

function draw_textbox(x,y,w,h,str,color,fontsize)
    annotation('textbox','Units','normalized','Position',[x y w h], ...
               'String',str,'Color',color,'FontSize',fontsize,'FontWeight','bold', ...
               'EdgeColor','none','Interpreter','none','HorizontalAlignment','center','VerticalAlignment','middle');
end


