function [z] = incLS(y,normP,monotonicDerivativeN,regularizeN)
%This function does an p-norm minimization of (z-y), subject to z being non-decreasing, with non-decreasing derivatives (i.e. concave) if requested. The returned vector z is a 'smoothed' version of y. This function is called by monoLS.
%%%INPUTS:
%y: column vector to smooth
%normP: norm used for minimization. Default normP=2 (least squares)
%monotonicDerivativeN: number of derivatives forced to be non-decreasing
% =1 means monotonic z & monotonic derivative (i.e. concave)
% =2 forces the second derivative to be monotonic
%regularizeN: number of samples that are force to be =0 for
%the last derivative forced by monotonicDerivativeFlag. These samples are
%taken at the end of the z. It avoids overfitting of said samples.
%%%%OUTPUT:
%z=Best-fit approximation of data given constraints
%Notice that a monotonic best-fit is always piece-wise constant (derivative is null almost everywhere) in presence
%of noise (i.e. if data is not truly monotonic). In the same way, if more
%derivatives are enforced, the last enforced one will be piece-wise
%constant, so one order less will be piece-wise linear, two orders less
%will be piece-wise quadratic and so forth.

%Same as monoLS, but assumes that ALL derivatives are forced to be
%non-negative. Only takes vector as argument (NOT matrix)

%TODO: if monotonicDerivativeN==0, then enforce regularization on BOTH
%edges of the data (problem is symmetric and can be overfit at both ends)
%TODO:
%1) Confim that the optimization (p=2) works & is fast (against some good optimization library?)
%4) Add optimization over both increasing and decreasing functions, and
%choose the best fit (still according to p-norm) over both.
%5) Fix convergence for p~=2 (?)
%% ARGUMENT CHECK:
if length(y)~=numel(y)
  error('incLS:dataFormat','Data (y) is not a vector')
end
y=reshape(y,length(y),1);
if nargin<2 || isempty(normP)
    normP=2;
end
if nargin<3 || isempty(monotonicDerivativeN)
    monotonicDerivativeN=0;
elseif monotonicDerivativeN>numel(y)
    error(['Cannot force the sign of ' num2str(monotonicDerivativeN) ' derivatives with only ' num2str(numel(y)) ' datapoints!'])
elseif monotonicDerivativeN>2
    error(['Forcing sign of ' num2str(monotonicDerivativeN+1) ' derivatives. Forcing more than the 3rd order derivative does not converge (although an optimal solution has to exist).'])
end

if nargin<4 || isempty(regularizeN) || monotonicDerivativeN==0
    %No regularization allowed if only one derivative is being forced, otherwise we may lose monotonicity
    regularizeN=0;
end

%% DO THE THING
    %Remove NaNs:
    [y2,idx]=removeNaN(y);

    %%Optimization
    %Get first guess:
    initDataGuess=defaultDataGuess(y2); %Default is line: admissible as long as the slope is non-neg

    %Construct matrix that computes data from optimized variables: (By recursion!)
    [A,w0]=getMatrix(numel(y2),monotonicDerivativeN,initDataGuess,regularizeN);

    %Now optimize:
    zz=optimize(A,y2,w0,normP);
    %zz=optimizeAlt(y2,normP,monotonicDerivativeN,initDataGuess,regularizeN);

    %Dealing with some ill-conditioned cases, in which a line is better than the solution found:
    if norm(zz-y2,normP)>norm(A*w0-y2,normP)
       zz=A*w0;
    end

    %Reconstructing data by adding the NaN values that were present
    z=nan(size(y));
    z(~idx)=zz;
end

%% AUXILIARY FUNCTIONS:
function [f,g,h]=cost(y,A,w,p)
    f=norm(y-A*w,p)^p;
    g=p*sign(A*w-y)'.*abs(A*w-y)'.^(p-1) *A;
    h=p*(p-1)*A'*A;
end

function z=defaultDataGuess(y,order)
    %First guess (init) for optimization target:
    x=[0:numel(y)-1]';
    pp=polyfit(x,y,1); %Fit a line to use as initial estimate: a line is always admissible!
    if pp(1)<0 %Flat line if best line is actually decreasing
      warning('Trying to fit monotonically increasing function, but data appears to be decreasing.')
        pp(1)=0
        pp(2)=mean(y);
    end
    z=pp(2)+pp(1)*x;
end

function [y,idx]=removeNaN(y)
    y=y(:); %Column vector
    idx=isnan(y);
    y=y(~idx);
end

function zz=optimize(A,y,w0,p)
    if p==2
        if A(end,3)<=A(end,2)
          %This solver is faster for order 0 and 1 (i.e. up to 2nd derivative sign constrained), does not converge for higher order
          [w,~,~,exitFlag]=lsqnonneg(A,y);
        else %Alternative solver, slower but better behaved:
          if ~exist('octave_config_info')
            opts=optimoptions('quadprog','Display','off');
          else
            opts=[];
          end
          B=A'*A;C=y'*A;
          [w,~,exitFlag]=quadprog(B,-C,[],[],[],[],zeros(size(w0)),[],w0,opts);
          %Note: also does not converge for order 3 and higher
        end
        if exitFlag<1
          warning('Optimization did not converge.')
        end
        zz=A*w;
    else %Generic solver for other norms, which result in non-quadratic programs (solver is slower, but somewhat better)
        %As of Mar 07 2017, this didn't work properly. Convergence?
        opts=optimoptions('fmincon','Display','off','SpecifyObjectiveGradient',true);
        w1=fmincon(@(x) cost(y,A,x,p),w0,[],[],[],[],zeros(size(w0)),[],[],opts);
        zz=A*w1;
    end
end

function [A,initSpaceGuess]=getMatrix(dataSize,order,initDataGuess,regularizeN)
    initSpaceGuess=[];
    %First, construct matrix that computes data from optimized variables: (By recursion!)
    A=tril(ones(dataSize));
    for i=1:order
        A(:,i+1:end)=fliplr(cumsum(fliplr(A(:,i+1:end)),2)); %Octave-compatible
    end
    if regularizeN~=0 %Forcing the value of the m-th derivative (m=monotonicDerivativeFlag+1),
       %which is the last constrained one, to be exactly 0 for the last n=regularizeFlag samples.
       %This avoids over-fitting to the first few datapoints (especially the
       %1st). %It is equivalent to reducing the size of the vector w() to be estimated.
       A(:,end-regularizeN+1:end)=[];
    end
    if nargin>2 && ~isempty(initDataGuess)
        w0=A\initDataGuess;
        initSpaceGuess=w0;
    end
end

function zz=optimizeAlt(y,p,order,initDataGuess,regN)
  %This is an alternative way to pose the problem. It avoids having an ill-conditioned matrix with very large entries when order>1. However, the feasible space becomes more convoluted, so the solver is much slower and does not converge sometimes because of MaxIter. It is good for finding approximately (but not strictly) monotonic solutions
  N=length(y);
  C=zeros(N);
  aux=[-1 1];
  C(1,1)=1;
  C(2,1:2)=aux;
  aux1=aux;
  m=2;
  for i=3:N
    if i<order+3
      aux1=conv(aux1,aux);
      m=m+1;
    end
    C(i,i-m+1:i)=aux1;
  end
    if p==2
        %Alternative solver:
        opts=optimoptions('quadprog','Display','off');
        [zz,~,exitFlag]=quadprog(eye(N),-y',-C,zeros(size(y)),[],[],[],[],initDataGuess,opts);
    else
      error('Unimplemented')
    end
end
