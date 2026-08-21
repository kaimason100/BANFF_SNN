function sys = make_dynamical_system(name, varargin)
%MAKE_DYNAMICAL_SYSTEM Return a struct describing a chosen dynamical system.
%   sys = make_dynamical_system(name)
%   sys = make_dynamical_system('custom', f_handle, dim, x0_default, params)
%
% Returns:
%   sys.name        : char
%   sys.dim         : D
%   sys.f           : @(t, x, params) -> dxdt  (returns SINGLE vector)
%   sys.x0_default  : [D x 1] single
%   sys.params      : struct (may be empty)
%
% Systems (mirroring labels in int_dyn, no extra scaling):
%   1D:  'Integrator', 'Pitchfork'
%   2D:  'Vanderpol', 'Hopf Normal Form'
%   3D:  'Rossler','Lorenz','Thomas',
%        'SprottA'..'SprottS' (SprottI = second variant in int_dyn),
%        'Chua1'..'Chua6', 'Rikitake','Nose Hoover','Halvorsen',
%        'MO0'..'MO15'
%
% Custom:
%   make_dynamical_system('custom', f_handle, dim, x0_default, params)

if nargin < 1
    error('System name is required.');
end

switch lower(name)

    %% 1D
    case 'integrator'
        D = 1;
        params = struct();
        if ~isempty(varargin), params = varargin{1}; end
        if ~isfield(params,'u') || ~isa(params.u,'function_handle')
            params.u = @(t) 0;
        end
        raw_f = @(t,x,p) p.u(t);
        x0 = single(0);  % not in sheet

    case 'pitchfork'
        D = 1; params = struct();
        raw_f = @(t,x,p) 0.5*x(1) - x(1).^3;
        x0 = single(0.1);  % from sheet

    %% 2D
case 'vanderpol'
    D = 2;

    % Parameters (FP32)
    params = struct( ...
        'mu', single(5) ...   % nonlinearity (>0 gives a stable limit cycle)
    );

    % Lienard/relaxation-coordinate Van der Pol form:
    %   x' = mu*(x - x^3/3 - y)
    %   y' = x/mu
    raw_f = @(t, x, p) [ p.mu*(x(1)-x(1).^3/3 - x(2));
                         x(1)/p.mu ];


    x0 = single([0.2; 0.3]);  % initial state


    case 'hopf normal form'
        D = 2; params = struct('beta',0.5,'sigma',-1);
        raw_f = @(t,x,p) [ p.beta*x(1) - x(2) + p.sigma*x(1).*(x(1).^2 + x(2).^2); ...
                            x(1) + p.beta*x(2) + p.sigma*x(2).*(x(1).^2 + x(2).^2) ];
        x0 = single([0.2; 0.3]);  % from sheet

    %% 3D classic
    case 'rossler'
        D = 3; params = struct('a',0.2,'b',0.2,'c',5.7);
        raw_f = @(t,x,p) [ -x(2) - x(3);  x(1) + p.a*x(2);  p.b + x(3).*(x(1) - p.c) ];
        x0 = single([0.0; 0.0; 0.0]);  % from sheet

    case 'lorenz'
        D = 3; params = struct('sigma',10,'rho',28,'beta',8/3);
        raw_f = @(t,x,p) [ p.sigma*(x(2)-x(1)); ...
                           x(1).*(p.rho - x(3)) - x(2); ...
                           x(1).*x(2) - p.beta*x(3) ];
        x0 = single([0.1; 0.1; 0.1]);  % from sheet

    case 'thomas'
        D = 3; params = struct('b',0.208186);
        raw_f = @(t,x,p) [ sin(x(2)) - p.b*x(1); ...
                           sin(x(3)) - p.b*x(2); ...
                           sin(x(1)) - p.b*x(3) ];
        x0 = single([-0.1; 0.1; 0.1]);  % from sheet

    %% Sprott A..S
    case 'sprotta'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2); -x(1)+x(2).*x(3); 1 - x(2).^2 ];
        x0=single([0.1; 0.2; 0.0]);  % from sheet

    case 'sprottb'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2).*x(3); x(1)-x(2); 1 - x(1).*x(2) ];
        x0=single([0.6; 0.0; 0.0]);  % from sheet

    case 'sprottc'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2).*x(3); x(1)-x(2); 1 - x(1).^2 ];
        x0=single([0.0; 0.5; 0.0]);  % from sheet

    case 'sprottd'
        D=3; params=struct();
        raw_f=@(t,x,p)[ -x(2); x(1)+x(3); x(1).*x(3) + 3*x(2).^2 ];
        x0=single([0.1; 0.1; 0.1]);  % from sheet

    case 'sprotte'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2).*x(3); x(1).^2 - x(2); 1 - 4*x(1) ];
        x0=single([-1.0; 1.0; 0.0]);  % from sheet

    case 'sprottf'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2)+x(3); -x(1)+0.5*x(2); x(1).^2 - x(3) ];
        x0=single([0.0; 0.2; 0.0]);  % from sheet

    case 'sprottg'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.4*x(1)+x(3); x(1).*x(3)-x(2); -x(1)+x(2) ];
        x0=single([0.1; 0.1; 0.1]);  % from sheet

    case 'sprotth'
        D=3; params=struct();
        raw_f=@(t,x,p)[ -x(2) + x(3).^2; x(1) + 0.5*x(2); x(1) - x(3) ];
        x0=single([0.1; 0.0; 0.0]);  % from sheet

    case 'sprotti'
        % Use the second variant from int_dyn: dX3 = x + y^2 - z
        D=3; params=struct();
        raw_f=@(t,x,p)[ -0.2*x(2); x(1)+x(3); x(1) + x(2).^2 - x(3) ];
        x0=single([0.0; 0.3; 0.0]);  % from sheet

    case 'sprottj'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 2*x(3); -2*x(2) + x(3); -x(1) + x(2) + x(2).^2 ];
        x0=single([-0.1; 1.0; 0.1]);  % from sheet

    case 'sprottk'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(1).*x(2) - x(3); x(1) - x(2); x(1) + 0.3*x(3) ];
        x0=single([1.0; 1.0; 2.0]);  % from sheet

    case 'sprottl'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2) + 3.9*x(3); 0.9*(x(1).^2) - x(2); 1 - x(1) ];
        x0=single([0.0; 12.0; -6.0]);  % from sheet

    case 'sprottm'
        D=3; params=struct();
        raw_f=@(t,x,p)[ -x(3); -x(1).^2 - x(2); 1.7 + 1.7*x(1) + x(2) ];
        x0=single([1.0; -0.8; 0.0]);  % from sheet

    case 'sprottn'
        D=3; params=struct();
        raw_f=@(t,x,p)[ -2*x(2); x(1) + x(3).^2; 1 + x(2) - 2*x(3) ];
        x0=single([4.5; 1.0; 0.0]);  % from sheet

    case 'sprotto'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2); x(1) - x(3); x(1) + x(1).*x(3) + 2.7*x(2) ];
        x0=single([0.0; 0.0; 0.5]);  % from sheet

    case 'sprottp'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 2.7*x(2) + x(3); -x(1) + x(2).^2; x(1) + x(2) ];
        x0=single([0.0; 0.3; 0.0]);  % from sheet

    case 'sprottq'
        D=3; params=struct();
        raw_f=@(t,x,p)[ -x(3); x(1) - x(2); 3.1*x(1) + x(2).^2 + 0.5*x(3) ];
        x0=single([1.0; 0.0; 0.0]);  % from sheet

    case 'sprottr'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.9 - x(2); 0.4 + x(3); x(1).*x(2) - x(3) ];
        x0=single([2.0; 0.0; 0.0]);  % from sheet

    case 'sprotts'
        D=3; params=struct();
        raw_f=@(t,x,p)[ -x(1) - 4*x(2); x(1) + x(3).^2; 1 + x(1) ];
        x0=single([0.0; 0.0; 1.0]);  % from sheet

    %% Chua 1..6
    case 'chua1'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.3*x(2) + x(1) - x(1).^3; x(1)+x(3); -x(2) ];
        x0=single([0.0; -3.0; 1.0]);  % from sheet

    case 'chua2'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.2*x(2) - x(1) + 2*tanh(x(1)); x(1)+x(3); -x(2) ];
        x0=single([0.0; 1.0; -6.0]);  % from sheet

    case 'chua3'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.2*x(2) + x(1) - x(1).*abs(x(1)); x(1)+x(3); -x(2) ];
        x0=single([0.0; 1.0; -3.0]);  % from sheet

    case 'chua4'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.2*x(2) - x(1) - 2*sin(x(1)); x(1)+x(3); -x(2) ];
        x0=single([0.0; 6.0; 0.0]);  % from sheet

    case 'chua5'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.2*x(2) - 0.3*x(1) + sign(x(1)); x(1)+x(3); -x(2) ];
        x0=single([0.0; 4.0; -8.0]);  % from sheet

    case 'chua6'
        D=3; params=struct();
        raw_f=@(t,x,p)[ 0.2*x(2) - x(1) + 2*atan(x(1)); x(1)+x(3); -x(2) ];
        x0=single([0.0; 7.6; 0.0]);  % from sheet

    %% Other 3D
    case 'rikitake'
        D=3; params=struct('mu',1,'alpha',1);
        raw_f=@(t,x,p)[ -p.mu*x(1) + x(2).*x(3); ...
                         -p.mu*x(2) + x(1).*(x(3)-p.alpha); ...
                          1 - x(1).*x(2) ];
        x0=single([0.0; 1.0; 0.0]);  % from sheet

    case 'nose hoover'
        D=3; params=struct();
        raw_f=@(t,x,p)[ x(2); x(2).*x(3) - x(1); 1 - x(2).^2 ];
        x0=single([0.5; 0.1; 0.1]);  % from sheet

    case 'halvorsen'
        % int_dyn used 'a' symbol without setting it; expose with default a=1
        D=3; params=struct('a',1);
        raw_f=@(t,x,p)[ -p.a*x(1) - 4*x(2) - 4*x(3) - x(2).^2; ...
                         -p.a*x(2) - 4*x(3) - 4*x(1) - x(3).^2; ...
                          p.a*x(3) - 4*x(1) - 4*x(2) - x(1).^2 ];
        x0=single([1; 1; 1]);  % not in sheet

    %% MO0..MO15 (jerk family)
    case 'mo0'
        D=3; params=struct('a',0.6,'b',1); g=@(x) abs(x)-1;
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; -0.7; 0.0]);  % from sheet

    case 'mo1'
        D=3; params=struct('a',0.6,'b',1); g=@(x) 1 - 6*max(x,0);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.0; 0.5]);  % from sheet

    case 'mo2'
        D=3; params=struct('a',0.6,'b',1); g=@(x) sign(x)-x;
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.0; 2.0]);  % from sheet

    case 'mo3'
        D=3; params=struct('a',1,'b',1); g=@(x) 1.1*(x.^2 - 1);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([1.0; 0.0; -1.0]);  % from sheet

    case 'mo4'
        D=3; params=struct('a',0.5,'b',1); g=@(x) x.*(x-1);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.1; 0.0]);  % from sheet

    case 'mo5'
        D=3; params=struct('a',0.7,'b',1); g=@(x) x.*(1-x.^2);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.0; 0.1]);  % from sheet

    case 'mo6'
        D=3; params=struct('a',0.4,'b',1); g=@(x) (x.^2).*(1-x);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.1; 0.2; 0.7]);  % from sheet

    case 'mo7'
        D=3; params=struct('a',0.6,'b',1); g=@(x) (x.^2).*(1-x.^2);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.0; 0.4]);  % from sheet

    case 'mo8'
        D=3; params=struct('a',0.5,'b',1); g=@(x) x.*(x.^4 - 1);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([-0.1; 0.1; 0.1]);  % from sheet

    case 'mo9'
        D=3; params=struct('a',0.4,'b',1); g=@(x) (x.^3).*(1-x);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([1.0; 0.2; 0.0]);  % from sheet

    case 'mo10'
        D=3; params=struct('a',0.6,'b',1); g=@(x) (x.^2).*(1-x.^3);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.01; 0.01; 0.01]);  % from sheet

    case 'mo11'
        D=3; params=struct('a',1,'b',1); g=@(x) 5 - exp(x);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 4.3; 0.0]);  % from sheet

    case 'mo12'
        D=3; params=struct('a',1,'b',1); g=@(x) 7 - 8*tanh(x);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 1.0; 6.0]);  % from sheet

    case 'mo13'
        D=3; params=struct('a',1,'b',1); g=@(x) 6*tanh(x) - 3*x;
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.0; 1.0]);  % from sheet

    case 'mo14'
        D=3; params=struct('a',0.6,'b',1); g=@(x) 6*atan(x) - x;
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 1.0; 6.0]);  % from sheet

    case 'mo15'
        D=3; params=struct('a',0.6,'b',1); g=@(x) x - 0.5*sinh(x);
        raw_f=@(t,x,p)[ x(2); x(3); g(x(1)) - p.a*x(3) - p.b*x(2) ];
        x0=single([0.0; 0.0; 1.0]);  % from sheet

    %% Custom
    case 'custom'
        if numel(varargin) < 4
            error(['For custom system, pass: f_handle, dim, x0_default, params. ' ...
                   'Example: make_dynamical_system(''custom'', @(t,x,p)[x(2); -x(1)], 2, [1;0], struct())']);
        end
        user_f = varargin{1};
        D      = varargin{2};
        x0     = single(varargin{3});
        params = varargin{4};
        if ~isa(user_f,'function_handle') || ~isscalar(D) || numel(x0)~=D
            error('Invalid custom arguments: check f_handle, dim, and x0_default.');
        end
        raw_f = user_f;

    otherwise
        error('Unknown system name: %s', name);
end

% Wrap to enforce single-precision output and column vector shape
f_single = @(t,x,p) single(raw_f(t, single(x), p));

sys = struct('name', name, ...
             'dim', D, ...
             'f', f_single, ...
             'x0_default', x0(:), ...
             'params', params);
end
