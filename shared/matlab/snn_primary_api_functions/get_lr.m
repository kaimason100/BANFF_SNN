% get_lr.m
function lr = get_lr(ep, epochs, sched)
t = (double(ep)-1)/max(1,double(epochs)-1);
switch lower(sched.type)
    case 'cosine'
        lr = sched.lr_end + 0.5*(sched.lr_start-sched.lr_end)*(1+cos(pi*t));
    case 'exponential'
        lr = sched.lr_start * (sched.lr_end/sched.lr_start)^t;
    otherwise
        error('Unknown learning-rate schedule "%s".', sched.type);
end
lr = single(lr);
end

