function [z] = monoLS(y,normP,derN,regN,oddSign,evenSign)
%This function does an p-norm minimization of (z-y), subject to z 
%(and optionally some of its derivatives, such as z',z'', etc.) being monotonic.
%The returned vector z is a non-parametrically smoothed version of y. 
%This function is a wrapper around incLS, which does all the numeric heavylifting.
%%%-----------------------INPUT:
%y: the column vector or matrix to smooth (monoLS acts along dim 1)
%p: norm used for minimization. Default p=2 (least squares)
%derN: order of the derivatives forced to be of constant sign.
%=0 means monotonic z
%=1 means monotonic z & monotonic first derivative (i.e. convex or concave function fitting)
%=2 forces the second derivative to be non-decreasing too, etc.
%regN: number of samples that are force to have a zero value for
%the last derivative forced by monotonicDerivativeFlag. These samples are
%taken at the end of the z if y is increasing, and at the beginning of z if
%it is decreasing. It avoids overfit of said samples.
%oddSign: the sign of the odd derivatives desired (positive=increasing, negative =decreasing, 0= the function will figure it out)
%evenSign: the sign of the even derivatives desired (positive= concave, opposite the oddSign = asymptoting function) %TODO: this sign should be given relative to the other one.
%%%%------------------------OUTPUT:
%z=Best-fit approximation of data given constraints
%Notice that a monotonic best-fit is always piece-wise constant (derivative is null almost everywhere) in presence
%of noise (i.e. if data is not truly monotonic). In the same way, if the sign of further
%derivatives are enforced, the last enforced one will be piece-wise
%constant, so one order less will be piece-wise linear, two orders less
%will be piece-wise quadratic and so forth.
%See also: incLS


%% ARGUMENT CHECK:
if nargin<2 || isempty(normP)
    normP=[];
end
if nargin<3 || isempty(derN) || derN==0
    derN=0; evenSign=0;
end
if nargin<4 || isempty(regN) || derN==0
    %No regularization allowed if only one derivative is being forced, otherwise we may lose monotonicity
    regN=[];
end
if numel(y)~=length(y) %More than 1 vector (matrix input, acting along columns)
    z=nan(size(y));
    for i=1:size(y,2)
        z(:,i)=monoLS(y(:,i),normP,derN,regN);
    end
else %Vector input-data
    y=reshape(y,length(y),1); %Column-vector
    if nargin<5 || isempty(oddSign) || oddSign==0
        %Determine if data is increasing or decreasing through corr sign:
        n=length(y);
        x=[0:n-1]';
        oddSign=sign(mean(x.*y(:))-((n-1)/2)*mean(y));
        %ALT: fit both possible signs and return the best:
        %if nargin<6 || isempty(evenSign) || evenSign==0
        %    evenSign=0;
        %end
        %[z1] = monoLS(y,normP,monotonicderN,regN,1,evenSign);
        %[z2] = monoLS(y,normP,monotonicderN,regN,-1,evenSign);
        %if sum((z1-y).^2)<sum((z2-y).^2)
        %  z=z1;
        %else
        %  z=z2;
        %end
        %return
    end
    if nargin<6 || isempty(evenSign) || evenSign==0
        %this forces asymptotic-like behavior as default (i.e. decaying exponentials rather exploding ones)
        evenSign=-1;
        %ALT: try both signs, choose the better-fitting one.
        %[z1] = monoLS(y,normP,monotonicderN,regN,oddSign,1);
        %[z2] = monoLS(y,normP,monotonicderN,regN,oddSign,-1);
        %if sum((z1-y).^2)<sum((z2-y).^2)
        %  z=z1;
        %else
        %  z=z2;
        %end
        %return
    end

    [y,a,s,f]=flipIfNeeded(y,oddSign,evenSign*oddSign); %Flip to fit a concave increasing function
    [z] = incLS(y,normP,derN,regN); %Find actual solution
    [z]=unflip(z,s,a,f); %Flip back
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
