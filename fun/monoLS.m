function [z] = monoLS(y,normP,monotonicDerivativeN,regularizeN,oddSign,evenSign)
%This function does an p-norm minimization of (z-y), subject to z being monotonic
%(or constant?). The returned vector z is a 'smoothed' version of y.
%INPUTS:
%y: the column vector or matrix to smooth (monoLS acts along dim 1)
%p: norm used for minimization. Default p=2 (least squares)
%monotonicDerivativeFlag: order of the derivatives forced to be of constant sign. 
%=0 means monotonic z (no derivatives forces)
%=1 means monotonic z & monotonic derivative(ie. convex or concave all the way)
%=2 forces the second derivative to be non-decreasing too, etc.
%regularizeFlag: number of samples that are force to have a zero value for
%the last derivative forced by monotonicDerivativeFlag. These samples are
%taken at the end of the z if y is increasing, and at the beginning of z if
%it is decreasing. It avoids overfit of said samples.
%oddSign: the sign of the odd derivatives desired (positive=increasing, negative =decreasing, 0= the function will figure it out)
%evenSign: the sign of the even derivatives desired (positive= concave)
%OUTPUT:
%z=Best-fit approximation of data given constraints
%Notice that a monotonic best-fit is always piece-wise constant (derivative is null almost everywhere) in presence
%of noise (i.e. if data is not truly monotonic). In the same way, if more
%derivatives are enforced, the last enforced one will be piece-wise
%constant, so one order less will be piece-wise linear, two orders less
%will be piece-wise quadratic and so forth.


%% ARGUMENT CHECK:
if nargin<2 || isempty(normP)
    normP=[];
end
if nargin<3 || isempty(monotonicDerivativeN)
    monotonicDerivativeN=0;
elseif monotonicDerivativeN>numel(y)
    error('Cannot force the sign of so many derivatives!')
end
if nargin<4 || isempty(regularizeN) || monotonicDerivativeN==0 
    %No regularization allowed if only one derivative is being forced, otherwise we may lose monotonicity
    regularizeN=[];
end
if nargin<5 || isempty(oddSign) || oddSign==0
    %Determine if data is increasing or decreasing through the best 2-norm line fit:
    %TODO: fit both increasing and decreasing functions, see which one is better
    %pp=polyfit([1:numel(y)]',y,1);
    %forceSign=sign(pp(1));
    oddSign=sign(nanmean(diff(y))); %Alternative determination of increasing/decreasing
    %s=sign(median(diff(y)));
end
if nargin<6 || isempty(evenSign) || evenSign==0
    evenSign=-oddSign; %this forces asymptotic-like behavior as default (i.e. decaying exponentials rather exploding ones)
end
if monotonicDerivativeN==0 %Regardless of everything else: no evenSign if no higher order derivatives are constrained in sign
    evenSign=0;
end

%%
if numel(y)~=length(y) %More than 1 vector (matrix input, acting along columns)
    z=nan(size(y));
    for i=1:size(y,2)
        z(:,i)=monoLS(y(:,i),normP,monotonicDerivativeN,regularizeN);
    end
else %Vector input-data
    [y,a,s,f]=flipIfNeeded(y,oddSign,evenSign);

    [z] = incLS(y,normP,monotonicDerivativeN,regularizeN);

    %Invert the flipping and positivization
    [z]=unflip(z,s,a,f);
end

end

function [z,a,s,f]=flipIfNeeded(y,oddSign,evenSign)
%Flips y as needed to get data that is positive, increasing and so on.
s=0; f=0;
if evenSign==0 || sign(evenSign)==sign(oddSign) %Neither convex nor concave OR all derivatives of the same sign
    if oddSign<0 %Want to fit decreasing
        y=-y;
        s=1;
    end
else %Even and odd derivatives are of different sign
    y=flipud(y); %Flipping the data indexing, which flips the sign of all odd derivatives
    f=1;
    if oddSign>0 %If we wanted increasing fit
        y=-y;
        s=1;
    end
end

a=min(y)-1;
z=y-a; %This ensures positive data
end

function [z]=unflip(y,s,a,f)
    z=y+a;
    if s==1
       z=-z;
    end
    if f==1
        z=flipud(z);
    end 
end