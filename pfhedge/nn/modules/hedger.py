from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam

# error: Skipping analyzing "tqdm": found module but no type hints or library stubs
from tqdm import tqdm  # type: ignore

from pfhedge._utils.hook import save_prev_output
from pfhedge._utils.lazy import has_lazy
from pfhedge._utils.operations import ensemble_mean
from pfhedge.features import get_feature
from pfhedge.instruments.base import Instrument
from pfhedge.instruments.derivative.base import Derivative
from pfhedge.instruments.primary.base import Primary

from .loss import EntropicRiskMeasure
from .loss import HedgeLoss

TensorOrFloat = Union[Tensor, float]


class Hedger(Module):
    """A ``torch.nn.Module`` to hedge and price derivatives.

    Args:
        model (torch.nn.Module): Hedging model to compute the hedge ratio at the
            next time step from the input features at the current time step.
            The input and output shapes should be :math:`(N, H_\\text{in})` and
            :math:`(N, 1)` respectively, where :math:`N` stands for the number simulated
            paths of the asset prices and :math:`H_\\text{in}` stands for the number of
            input features (namely, ``len(inputs)``).
        inputs (list[str|Feature]): List of (names of) input features to feed to model.
            See ``[str(f) for f in pfhedge.features.FEATURES]`` for valid options.
        criterion (HedgeLoss, default=EntropicRiskMeasure()):
            Loss function to minimize by hedging.
            Default: :class:`pfhedge.nn.EntropicRiskMeasure()` .

    Shape:
        - Input: :math:`(N, H_{\\text{in}})` where :math:`H_{\\text{in}}` is
          the number of input features.
        - Output: :math:`(N, 1)`

    Examples:

        A hedger that uses Black-Scholes' delta hedging strategy.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import BlackScholes
        >>> from pfhedge.nn import Hedger
        >>>
        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> model = BlackScholes(derivative)
        >>> hedger = Hedger(model, model.inputs())
        >>> hedger
        Hedger(
          inputs=['log_moneyness', 'expiry_time', 'volatility']
          (model): BSEuropeanOption()
          (criterion): EntropicRiskMeasure()
        )

        A hedger that uses Whalley-Wilmott's no-transaction-band strategy.

        >>> from pfhedge.nn import WhalleyWilmott
        >>>
        >>> model = WhalleyWilmott(derivative)
        >>> hedger = Hedger(model, model.inputs())
        >>> hedger
        Hedger(
          inputs=['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
          (model): WhalleyWilmott(
            (bs): BSEuropeanOption()
            (clamp): Clamp()
          )
          (criterion): EntropicRiskMeasure()
        )

        A hedger that takes naked positions (never hedge at all).

        >>> from pfhedge.nn import Naked
        >>>
        >>> hedger = Hedger(Naked(), ["empty"])

        A hedger represented by a neural network (Deep Hedging).

        >>> from pfhedge.nn import MultiLayerPerceptron
        >>>
        >>> model = MultiLayerPerceptron()
        >>> hedger = Hedger(model, ["moneyness", "expiry_time", "volatility"])
        >>> _ = hedger.compute_pnl(derivative, n_paths=1)  # Lazily materialize
        >>> hedger
        Hedger(
          inputs=['moneyness', 'expiry_time', 'volatility']
          (model): MultiLayerPerceptron(
            (0): Linear(in_features=3, out_features=32, bias=True)
            (1): ReLU()
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): ReLU()
            (4): Linear(in_features=32, out_features=32, bias=True)
            (5): ReLU()
            (6): Linear(in_features=32, out_features=32, bias=True)
            (7): ReLU()
            (8): Linear(in_features=32, out_features=1, bias=True)
            (9): Identity()
          )
          (criterion): EntropicRiskMeasure()
        )
        >>> history = hedger.fit(derivative, verbose=False, n_paths=1, n_epochs=1)
        >>> hedger.price(derivative)
        tensor(...)

        It is possible to hedge a derivative with another listed derivative by
        ``Derivative.list()`` method.

        >>> from pfhedge.instruments import LookbackOption
        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> pricer = lambda derivative: BlackScholes(derivative).price(
        ...     log_moneyness=derivative.log_moneyness(),
        ...     expiry_time=derivative.time_to_maturity(),
        ...     volatility=derivative.ul().volatility)
        >>>
        >>> stock = BrownianStock()
        >>> hedging_instrument = EuropeanOption(stock, maturity=5/250)
        >>> hedging_instrument.list(pricer, cost=1e-4)
        >>> derivative = LookbackOption(stock)
        >>> derivative.simulate()
        >>>
        >>> stock = BrownianStock()
        >>> hedger = Hedger(
        ...     MultiLayerPerceptron(),
        ...     inputs=["moneyness", "expiry_time", "volatility"])
        >>> _ = hedger.fit(
        ...     derivative,
        ...     hedge=hedging_instrument,
        ...     verbose=False,
        ...     n_paths=1,
        ...     n_epochs=1)
        >>> hedger.price(derivative)
        tensor(...)
    """

    def __init__(
        self,
        model: Module,
        inputs: List[str],
        criterion: HedgeLoss = EntropicRiskMeasure(),
    ):
        super().__init__()

        self.model = model
        self.inputs = [get_feature(i) for i in inputs]
        self.criterion = criterion

        self.register_forward_hook(save_prev_output)

    def forward(self, input: Tensor) -> Tensor:
        """Returns the outout of ``self.model``.

        The output represents the hedge ratio at the next time step.
        """
        return self.model(input)

    def extra_repr(self) -> str:
        return "inputs=" + str(list(map(str, self.inputs)))

    def compute_pnl(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_paths: int = 1000,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
    ) -> Tensor:
        """Returns the profit and loss distribution after hedging.

        A hedger sells the derivative to its customer and obliges to settle the payoff
        at maturity. The dealer hedges the risk of this liability by trading
        the underlying instrument of the derivative based on ``self.model``.
        The resulting profit and loss is obtained by adding up the payoff to the
        customer, capital gains from the underlying asset, and the transaction cost.

        Args:
            derivative (pfhedge.instruments.Derivative): The derivative to hedge.
            hedge (pfhedge.instruments.Instrument, optional): The instrument to hedge
                the risk of ``derivative``.
                If ``None`` (default), use ``derivative.ul()``.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the underlying instrument of the derivative.
                If ``None`` (default), it uses the default value.

        Shape:
            - Output: :math:`(N)`, where :math:`N` is the number of paths.

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>> from pfhedge.nn import Hedger
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.compute_pnl(derivative, n_paths=2)
            tensor([..., ...])
        """
        self.inputs = [f.of(derivative, self) for f in self.inputs]
        hedge = hedge if hedge is not None else derivative.ul()
        hedge = cast(Union[Primary, Derivative], hedge)

        derivative.simulate(n_paths=n_paths, init_state=init_state)
        # cashflow: shape (N, T - 1)
        cashflow = hedge.spot.diff(dim=-1)

        # prev_output: shape (N)
        save_prev_output(self, None, torch.zeros_like(hedge.spot[..., :1]))
        pnl = torch.zeros_like(hedge.spot[..., 0])

        # Simulate hedging over time.
        for i in range(hedge.spot.size(1) - 1):
            prev_output = self.get_buffer("prev_output").reshape(-1)

            # Compute the hedge ratio at the next time step.
            output = self(torch.cat([f[i] for f in self.inputs], 1)).reshape(-1)

            # Receive profit and loss from the underlying asset.
            pnl += output * cashflow[..., i]

            # Deduct transaction cost.
            pnl -= hedge.cost * (output - prev_output).abs() * hedge.spot[..., i]

        # Settle the derivative's payoff.
        pnl -= derivative.payoff()

        # Delete the attribute `prev_output` in case a hedger has a feature
        # `PrevHedge` and one calls `compute_pnl` twice.
        # If `prev_output` is not deleted, `prev_output` at the last time step
        # in the first call could be referred to by `PrevHedge` at the first
        # time step at the second call,
        # which results in an unexpected output (while we expect zeros).
        if hasattr(self, "prev_output"):
            delattr(self, "prev_output")

        return pnl

    def compute_loss(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
        enable_grad: bool = True,
    ) -> Tensor:
        """Returns the loss of the profit and loss distribution after hedging.

        Args:
            derivative (pfhedge.instruments.Derivative): The derivative to hedge.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            n_times (int, default=1): If ``n_times > 1``, returns the ensemble mean
                of the losses computed through multiple simulations.
            init_state (tuple, optional): The initial price of the underlying
                instrument of the derivative.
                If ``None`` (default), sensible default value is used.
            enable_grad (bool, default=True): Context-manager that sets gradient
                calculation to on or off.

        Shape:
            - Output: :math:`()`

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>> from pfhedge.nn import Hedger
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.compute_loss(derivative, n_paths=2)
            tensor(...)
        """
        with torch.set_grad_enabled(enable_grad):
            loss = lambda: self.criterion(
                self.compute_pnl(
                    derivative, hedge=hedge, n_paths=n_paths, init_state=init_state
                )
            )
            mean_loss = ensemble_mean(loss, n_times=n_times)

        return mean_loss

    def fit(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_epochs: int = 100,
        n_paths: int = 1000,
        n_times: int = 1,
        optimizer=Adam,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
        verbose: bool = True,
    ) -> List[float]:
        """Train the hedging model to hedge the given derivative.

        It returns the trade history, that is, validation loss after each simulation.

        Args:
            derivative (pfhedge.instruments.Derivative): The derivative to hedge.
            n_epochs (int, default=100): Number of Monte-Carlo simulations.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            n_times (int, default=1): If ``n_times > 1``, returns the ensemble mean of
                the losses computed through multiple simulations.
            optimizer (torch.optim.Optimizer, default=Adam): The optimizer algorithm
                to use.  It can be an instance or a class of ``torch.optim.Optimizer``.
            init_state (tuple, optional): The initial price of the underlying
                instrument of the derivative.
                If ``None`` (default), sensible default value is used.
            verbose (bool, default=True): If ``True``, print progress of the training to
                standard output.

        Returns:
            list[float]

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import MultiLayerPerceptron
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = MultiLayerPerceptron()
            >>> hedger = Hedger(model, ["moneyness", "expiry_time", "volatility"])
            >>> history = hedger.fit(derivative, verbose=False, n_paths=1, n_epochs=1)

            One can use a custom optimizer as follows.

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import MultiLayerPerceptron
            >>> from torch.optim import SGD
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> hedger = Hedger(MultiLayerPerceptron(), ["empty"])
            >>> # Run a placeholder forward to initialize lazy parameters
            >>> _ = hedger.compute_pnl(derivative, n_paths=1)
            >>> _ = hedger.fit(
            ...     derivative,
            ...     optimizer=SGD(hedger.parameters(), lr=0.1),
            ...     n_epochs=1,
            ...     verbose=False)

            One can also pass a class object of an optimizer.

            >>> from torch.optim import Adadelta
            >>> derivative = EuropeanOption(BrownianStock())
            >>>
            >>> hedger = Hedger(MultiLayerPerceptron(), ["empty"])
            >>> _ = hedger.fit(
            ...     derivative,
            ...     optimizer=Adadelta,
            ...     n_epochs=1,
            ...     verbose=False)
        """
        if isinstance(optimizer, type):
            if has_lazy(self):
                # Run a placeholder forward to initialize lazy parameters
                _ = self.compute_pnl(derivative, n_paths=1)
            optimizer = optimizer(self.model.parameters())
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer is not torch.optim.Optimizer")

        def compute_loss(**kwargs) -> Tensor:
            return self.compute_loss(
                derivative,
                hedge=hedge,
                n_paths=n_paths,
                init_state=init_state,
                **kwargs,
            )

        history = []
        progress = tqdm(range(n_epochs), disable=not verbose)
        for _ in progress:
            # Compute training loss and backpropagate
            self.train()
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()

            # Compute validation loss
            self.eval()
            loss = compute_loss(n_times=n_times, enable_grad=False)
            history.append(loss.item())

            if verbose:
                progress.desc = f"Loss={loss:.5e}"

        return history

    def price(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
        enable_grad: bool = False,
    ) -> Tensor:
        """Evaluate the premium of the given derivative.

        Args:
            derivative (pfhedge.instuments.Derivative): The derivative to price.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            n_times (int, default=1): If ``n_times > 1``, returns the ensemble mean of
                the losses computed through multiple simulations.
            init_state (tuple, optional): The initial price of the underlying
                instrument of the derivative.
                If ``None`` (default), sensible default value is used.
            enable_grad (bool, default=False): Context-manager that sets gradient
                calculation to on or off.

        Shape:
            - Output: :math:`()`

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>> from pfhedge.nn import Hedger
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.price(derivative, n_paths=2)
            tensor(...)
        """
        with torch.set_grad_enabled(enable_grad):
            # Negative because selling
            pricer = lambda: -self.criterion.cash(
                self.compute_pnl(
                    derivative, hedge=hedge, n_paths=n_paths, init_state=init_state
                )
            )
            mean_price = ensemble_mean(pricer, n_times=n_times)

        return mean_price