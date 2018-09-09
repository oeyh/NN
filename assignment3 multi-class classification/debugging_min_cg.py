
def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    if gfk is None:
        gfk = fprime(xk)

    if isinstance(fprime, tuple):
        eps = fprime[1]
        fprime = fprime[0]
        newargs = (f, eps) + args
        gradient = False
    else:
        newargs = args
        gradient = True

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s*pk, *newargs)
        if gradient:
            gc[0] += 1
        else:
            fc[0] += len(xk) + 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
        phi, derphi, old_fval, old_old_fval, derphi0,
        c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe2(phi, derphi=None, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None,
                         extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable f(x)
        Objective scalar function.
    derphi : callable f'(x), optional
        Objective function derivative (can be None)
    phi0 : float, optional
        Value of phi at s=0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value of derphi at s=0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform
    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star
    phi0 : float
        phi at 0
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.
    For the zoom phase it uses an algorithm by [...].
    """

    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None and derphi is not None:
        derphi0 = derphi(0.)

    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    phi_a1 = phi(alpha1)
    # derphi_a1 = derphi(alpha1)  evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    if extra_condition is None:
        def extra_condition(alpha, phi): return True

    for i in xrange(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None

            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = "The line search algorithm could not find a solution " + \
                      "less than or equal to amax: %s" % amax

            warn(msg, LineSearchWarning)
            break

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha0, alpha1, phi_a0,
                      phi_a1, derphi_a0, phi, derphi,
                      phi0, derphi0, c1, c2, extra_condition)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha1, alpha0, phi_a1,
                      phi_a0, derphi_a1, phi, derphi,
                      phi0, derphi0, c1, c2, extra_condition)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn('The line search algorithm did not converge', LineSearchWarning)

    return alpha_star, phi_star, phi0, derphi_star


def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
                       extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.
    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform
    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.
    For the zoom phase it uses an algorithm by [...].
    """
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha * pk, *args)

    if isinstance(myfprime, tuple):
        def derphi(alpha):
            fc[0] += len(xk) + 1
            eps = myfprime[1]
            fprime = myfprime[0]
            newargs = (f, eps) + args
            gval[0] = fprime(xk + alpha * pk, *newargs)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0], pk)
    else:
        fprime = myfprime

        def derphi(alpha):
            gc[0] += 1
            gval[0] = fprime(xk + alpha * pk, *args)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0], pk)

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)

    if extra_condition is not None:
        # Add the current gradient as argument, to avoid needless
        # re-evaluation
        def extra_condition2(alpha, phi):
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
        phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
        extra_condition2, maxiter=maxiter)

    if derphi_star is None:
        warn('The line search algorithm did not converge', LineSearchWarning)
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def _minimize_cg(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                 disp=False, return_all=False,
                 **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    """
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    xk = x0

    # Sets the initial step guess to dx ~ 1
    old_fval = f(xk)
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, ord=norm)

    sigma_3 = 0.01

    while (gnorm > gtol) and (k < maxiter):
        deltak = numpy.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, numpy.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            #
            # See Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step

            # Accept step if it leads to convergence.
            if gnorm <= gtol:
                return True

            # Accept step if sufficient descent condition applies.
            return numpy.dot(pk, gfk) <= -sigma_3 * numpy.dot(gfk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
                                     old_old_fval, c2=0.4, amin=1e-100, amax=1e100,
                                     extra_condition=descent_condition)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)

        if retall:
            allvecs.append(xk)
        if callback is not None:
            callback(xk)
        k += 1

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % func_calls[0])
        print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval, jac=gfk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result
