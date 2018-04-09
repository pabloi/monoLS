function [z] = monoLS(y,normP,monotonicDerivativeN,regularizeN,oddSign,evenSign)
%This function does an p-norm minimization of (z-y), subject to z being monotonic
%(or constant?). The returned vector z is a 'smoothed' version of y.
%INPUTS:
%y: the column vector or matrix to smooth (monoLS acts along dim 1)
%p: norm used for minimization. Default p=2 (least squares)
%monotonicDerivativeFlag: order of the derivatives forced to be of constant sign. =0 means
%monotonic z, =1 means monotonic z & monotonic derivative (e.g. concave or
%convex all the way), =2 forces the second derivative to be of constant
%sign too, etc.
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
    normP=2;
end
if nargin<3 || isempty(monotonicDerivativeN)
    monotonicDerivativeN=0;
elseif monotonicDerivativeN>numel(y)
    error('Cannot force the sign of so many derivatives!')
end
if nargin<4 || isempty(regularizeN) || monotonicDerivativeN==0 
    %No regularization allowed if only one derivative is being forced, otherwise we may lose monotonicity
    regularizeN=0;
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
   evenSign=NaN; %Doxy 
end

%%
if numel(y)~=length(y) %More than 1 vector (matrix input, acting along columns)
    z=nan(size(y));
    for i=1:size(y,2)
        z(:,i)=monoLS(y(:,i),normP,monotonicDerivativeN,regularizeN);
    end
else %Vector input-data
    [y,a,s]=flipIfNeeded(y,oddSign,evenSign);

    [z] = incLS(y,normP,monotonicDerivativeN,regularizeN);

    %Invert the flipping and positivization
    [z]=unflip(z,s,a);
end

end

function [z,a,s]=flipIfNeeded(y,oddSign,evenSign)
%Flips y as needed to get data that is positive, increasing and so on.

%To make it simple, we flip the data so that it is always increasing & positive
if oddSign>0
    y=-y; %Data is now decreasing, f'<0
end
s=oddSign;
a=min(y)-1;
y=y-a; %Data is now f>0 & f'<0
z=flipud(y); %Data is now f>0, f'>0, as I inverted the 'x' axis
end

function [z]=unflip(y,s,a)
    %Invert the flipping and positivization
    z=flipud(y);
    z=z+a;
    if s>0
        z=-z;
    end
end