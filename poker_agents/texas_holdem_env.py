"""Texas Hold'em Poker Gymnasium Environment.

This module implements a Texas Hold'em poker environment with limit betting,
tournament format, and minimal state representation.
"""

from __future__ import annotations

import enum
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(enum.IntEnum):
    """Poker actions in limit betting."""

    FOLD = 0
    CALL = 1
    RAISE = 2


class HandRank(enum.IntEnum):
    """Poker hand rankings from lowest to highest."""

    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


@dataclass
class Card:
    """Represents a playing card."""

    suit: int  # 0=clubs, 1=diamonds, 2=hearts, 3=spades
    rank: int  # 0=2, 1=3, ..., 8=10, 9=J, 10=Q, 11=K, 12=A

    def __str__(self) -> str:
        suits = ["♣", "♦", "♥", "♠"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return f"{ranks[self.rank]}{suits[self.suit]}"


@dataclass
class Player:
    """Represents a player in the game."""

    player_id: int
    stack: int
    hand: list[Card] = field(default_factory=list)
    is_active: bool = True
    is_all_in: bool = False
    current_bet: int = 0
    total_contributed: int = 0

    def reset_hand(self) -> None:
        """Reset player state for a new hand."""
        self.hand = []
        self.current_bet = 0
        self.total_contributed = 0
        self.is_all_in = False


class HandEvaluator:
    """Evaluates poker hands."""

    @staticmethod
    def evaluate_hand(cards: Sequence[Card]) -> tuple[HandRank, list[int]]:
        """Evaluate a hand and return (rank, tiebreaker values).

        Args:
            cards: List of 5 or more cards

        Returns:
            Tuple of (hand rank, tiebreaker values for comparison)
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate a hand")

        # Get best 5-card hand from available cards
        best_rank = HandRank.HIGH_CARD
        best_tiebreakers: list[int] = []

        # Try all combinations of 5 cards
        from itertools import combinations

        for combo in combinations(cards, 5):
            rank, tiebreakers = HandEvaluator._evaluate_five_cards(list(combo))
            if rank > best_rank or (
                rank == best_rank and tiebreakers > best_tiebreakers
            ):
                best_rank = rank
                best_tiebreakers = tiebreakers

        return best_rank, best_tiebreakers

    @staticmethod
    def _evaluate_five_cards(cards: list[Card]) -> tuple[HandRank, list[int]]:
        """Evaluate exactly 5 cards."""
        # Sort by rank
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        ranks = [c.rank for c in sorted_cards]
        suits = [c.suit for c in sorted_cards]

        # Count rank frequencies
        rank_counts: dict[int, int] = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(
            rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True
        )

        # Check for flush
        is_flush = len(set(suits)) == 1

        # Check for straight
        is_straight = False
        straight_high = 0
        # Check normal straight
        if len(set(ranks)) == 5:
            if ranks[0] - ranks[4] == 4:
                is_straight = True
                straight_high = ranks[0]
            # Check A-2-3-4-5 straight (wheel)
            elif ranks == [12, 3, 2, 1, 0]:
                is_straight = True
                straight_high = 3  # 5-high straight

        # Royal flush
        if is_straight and is_flush and straight_high == 12:
            return HandRank.ROYAL_FLUSH, []

        # Straight flush
        if is_straight and is_flush:
            return HandRank.STRAIGHT_FLUSH, [straight_high]

        # Four of a kind
        if counts == [4, 1]:
            four_rank = unique_ranks[0]
            kicker = unique_ranks[1]
            return HandRank.FOUR_OF_A_KIND, [four_rank, kicker]

        # Full house
        if counts == [3, 2]:
            three_rank = unique_ranks[0]
            pair_rank = unique_ranks[1]
            return HandRank.FULL_HOUSE, [three_rank, pair_rank]

        # Flush
        if is_flush:
            return HandRank.FLUSH, ranks

        # Straight
        if is_straight:
            return HandRank.STRAIGHT, [straight_high]

        # Three of a kind
        if counts == [3, 1, 1]:
            three_rank = unique_ranks[0]
            kickers = sorted(unique_ranks[1:], reverse=True)
            return HandRank.THREE_OF_A_KIND, [three_rank] + kickers

        # Two pair
        if counts == [2, 2, 1]:
            pairs = sorted([unique_ranks[0], unique_ranks[1]], reverse=True)
            kicker = unique_ranks[2]
            return HandRank.TWO_PAIR, pairs + [kicker]

        # Pair
        if counts == [2, 1, 1, 1]:
            pair_rank = unique_ranks[0]
            kickers = sorted(unique_ranks[1:], reverse=True)
            return HandRank.PAIR, [pair_rank] + kickers

        # High card
        return HandRank.HIGH_CARD, ranks


class TexasHoldemEnv(gym.Env):
    """Texas Hold'em Poker Environment with limit betting.

    This environment implements a tournament-style Texas Hold'em game with:
    - Limit betting (fold/call/raise)
    - Fixed blinds and stack sizes
    - Elimination tournament format
    - Minimal state representation
    - Self-play mode: agent controls all players

    In self-play mode, the agent provides actions for whichever player is
    currently acting. The observation is always from the current player's
    perspective, allowing the agent to learn from all positions.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_players: int = 4,
        initial_stack: int = 1000,
        small_blind: int = 10,
        big_blind: int = 20,
        raise_amount: int = 20,
        max_raises_per_round: int = 3,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        """Initialize the poker environment.

        Args:
            num_players: Number of players (2-4)
            initial_stack: Starting stack size for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
            raise_amount: Fixed raise amount in limit betting
            max_raises_per_round: Maximum raises allowed per betting round
            seed: Random seed for reproducibility
            render_mode: Render mode ("human" or "ansi")
        """
        if not 2 <= num_players <= 4:
            raise ValueError("num_players must be between 2 and 4")

        self.num_players = num_players
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.raise_amount = raise_amount
        self.max_raises_per_round = max_raises_per_round
        self.render_mode = render_mode

        # Action space: fold, call, raise
        self.action_space = spaces.Discrete(3)

        # Observation space: minimal state representation
        # [player_stack, player_hand_rank, player_hand_high, community_cards_mask,
        #  pot_size, current_bet_to_call, position, num_active_players, is_preflop,
        #  first_player_idx]
        # Using float32 for compatibility
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        # Game state
        self.players: list[Player] = []
        self.community_cards: list[Card] = []
        self.deck: list[Card] = []
        self.pot: int = 0
        self.current_player_idx: int = 0
        self.dealer_idx: int = 0
        self.betting_round: int = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.raises_this_round: int = 0
        self.last_raise_player: int = -1
        self.players_acted_this_round: set[int] = (
            set()
        )  # Track which players have acted
        self.hand_evaluator = HandEvaluator()

        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Observation and info dict
        """
        if seed is not None:
            self.rng.seed(seed)
            random.seed(seed)

        # Initialize players
        self.players = [
            Player(player_id=i, stack=self.initial_stack)
            for i in range(self.num_players)
        ]

        # Start a new hand
        self._start_new_hand()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        In self-play mode, the agent controls all players. Each step requires
        an action for the current player (whoever is acting).

        Args:
            action: Action to take (0=fold, 1=call, 2=raise) for the current player

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            - observation: From the perspective of the next player to act (or current if game ended)
            - reward: Reward for player 0 (the learning agent) only
            - terminated: Whether the tournament has ended
            - truncated: Always False (never truncate - tournament ends naturally)
            - info: Additional information about the game state
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        reward = 0.0
        terminated = False
        truncated = False  # Never truncate - tournament ends naturally when only one player remains

        current_player = self.players[self.current_player_idx]

        # Execute action for the current player
        if action == Action.FOLD:
            current_player.is_active = False
        elif action == Action.CALL:
            call_amount = min(
                self._get_current_bet_to_call(),
                current_player.stack,
            )
            current_player.stack -= call_amount
            current_player.current_bet += call_amount
            current_player.total_contributed += call_amount
            self.pot += call_amount
            if current_player.stack == 0:
                current_player.is_all_in = True
        elif action == Action.RAISE:
            if self.raises_this_round >= self.max_raises_per_round:
                # Can't raise, treat as call
                call_amount = min(
                    self._get_current_bet_to_call(),
                    current_player.stack,
                )
                current_player.stack -= call_amount
                current_player.current_bet += call_amount
                current_player.total_contributed += call_amount
                self.pot += call_amount
            else:
                raise_total = min(
                    self._get_current_bet_to_call() + self.raise_amount,
                    current_player.stack,
                )
                current_player.stack -= raise_total
                current_player.current_bet += raise_total
                current_player.total_contributed += raise_total
                self.pot += raise_total
                self.raises_this_round += 1
                self.last_raise_player = self.current_player_idx
                if current_player.stack == 0:
                    current_player.is_all_in = True

        # Track that this player has acted
        self.players_acted_this_round.add(self.current_player_idx)

        # Move to next player
        self._advance_to_next_player()

        # Check if betting round is complete
        if self._is_betting_round_complete():
            # Move to next betting round or showdown
            if self.betting_round < 3:
                self.betting_round += 1
                if self.betting_round == 1:  # Flop
                    self._deal_community_cards(3)
                elif self.betting_round == 2:  # Turn
                    self._deal_community_cards(1)
                elif self.betting_round == 3:  # River
                    self._deal_community_cards(1)

                # Reset betting state
                self._reset_betting_round()
            else:
                # Showdown - determine winner
                winners = self._determine_winners()
                if len(winners) == 1:
                    winner = winners[0]
                    winner.stack += self.pot
                    reward = self.pot if winner.player_id == 0 else 0.0
                else:
                    # Split pot
                    split_amount = self.pot // len(winners)
                    for winner in winners:
                        winner.stack += split_amount
                    if any(w.player_id == 0 for w in winners):
                        reward = split_amount

                # Remove eliminated players (stack == 0)
                self.players = [p for p in self.players if p.stack > 0]

                # Check for tournament end - only one player left
                if len(self.players) <= 1:
                    terminated = True
                else:
                    # Rotate dealer and start new hand
                    self.dealer_idx = (self.dealer_idx + 1) % len(self.players)
                    self._start_new_hand()

        # Check if only one player remains (folded or eliminated)
        # Remove eliminated players first
        self.players = [p for p in self.players if p.stack > 0]

        # Check for tournament end - only one player left
        if len(self.players) <= 1:
            terminated = True
        else:
            # Check if only one active player remains (others folded)
            active_players = [p for p in self.players if p.is_active]
            if len(active_players) == 1:
                winner = active_players[0]
                winner.stack += self.pot
                reward = self.pot if winner.player_id == 0 else 0.0
                terminated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _start_new_hand(self) -> None:
        """Start a new hand."""
        # Remove eliminated players (stack == 0)
        self.players = [p for p in self.players if p.stack > 0]

        # Tournament ends if less than 2 players remain
        if len(self.players) < 2:
            return

        # Reset player states
        for player in self.players:
            player.reset_hand()
            player.is_active = True

        # Create and shuffle deck
        self.deck = [Card(suit=s, rank=r) for s in range(4) for r in range(13)]
        random.shuffle(self.deck)

        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                player.hand.append(self.deck.pop())

        # Initialize betting round
        self.community_cards = []
        self.betting_round = 0
        self.pot = 0  # Reset pot before posting blinds

        # Post blinds (this will add to the pot)
        self._post_blinds()

        self._reset_betting_round()

        # Set current player (first to act after big blind)
        self.current_player_idx = (self.dealer_idx + 3) % len(self.players)

    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        if len(self.players) < 2:
            return

        small_blind_idx = (self.dealer_idx + 1) % len(self.players)
        big_blind_idx = (self.dealer_idx + 2) % len(self.players)

        # Small blind
        sb_player = self.players[small_blind_idx]
        sb_amount = min(self.small_blind, sb_player.stack)
        sb_player.stack -= sb_amount
        sb_player.current_bet = sb_amount
        sb_player.total_contributed = sb_amount
        self.pot += sb_amount
        if sb_player.stack == 0:
            sb_player.is_all_in = True

        # Big blind
        bb_player = self.players[big_blind_idx]
        bb_amount = min(self.big_blind, bb_player.stack)
        bb_player.stack -= bb_amount
        bb_player.current_bet = bb_amount
        bb_player.total_contributed = bb_amount
        self.pot += bb_amount
        if bb_player.stack == 0:
            bb_player.is_all_in = True

    def _reset_betting_round(self) -> None:
        """Reset betting state for a new round."""
        self.raises_this_round = 0
        self.last_raise_player = -1
        self.players_acted_this_round = set()  # Reset tracking of who has acted

        # Find first active player to act
        # In preflop, first to act is after big blind
        # In later rounds, first to act is after dealer
        if self.betting_round == 0:
            self.current_player_idx = (self.dealer_idx + 3) % len(self.players)
        else:
            self.current_player_idx = (self.dealer_idx + 1) % len(self.players)

        # Skip inactive players
        while not self.players[self.current_player_idx].is_active:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

        # Reset current bets for new round
        # In preflop, don't reset bets because blinds have already been posted
        if self.betting_round > 0:
            for player in self.players:
                player.current_bet = 0

    def _deal_community_cards(self, num_cards: int) -> None:
        """Deal community cards."""
        for _ in range(num_cards):
            self.community_cards.append(self.deck.pop())

    def _advance_to_next_player(self) -> None:
        """Move to the next active player."""
        start_idx = self.current_player_idx
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

            # Prevent infinite loop
            if self.current_player_idx == start_idx:
                break

            player = self.players[self.current_player_idx]
            if player.is_active and player.stack > 0:
                # Check if player needs to act (hasn't matched the bet or can raise)
                max_bet = self._get_max_bet()
                needs_to_act = player.current_bet < max_bet
                can_raise = (
                    self.raises_this_round < self.max_raises_per_round
                    and self.current_player_idx != self.last_raise_player
                    and player.stack > max_bet - player.current_bet
                )
                if needs_to_act or can_raise:
                    break

    def _get_current_bet_to_call(self) -> int:
        """Get the amount needed to call."""
        max_bet = self._get_max_bet()
        current_player = self.players[self.current_player_idx]
        return max(0, max_bet - current_player.current_bet)

    def _get_max_bet(self) -> int:
        """Get the maximum bet in the current round."""
        return max((p.current_bet for p in self.players), default=0)

    def _is_betting_round_complete(self) -> bool:
        """Check if the betting round is complete."""
        # All active players have matched the bet or are all-in
        max_bet = self._get_max_bet()
        active_players = [p for p in self.players if p.is_active and p.stack > 0]
        active_player_ids = {
            i for i, p in enumerate(self.players) if p.is_active and p.stack > 0
        }

        if len(active_players) <= 1:
            return True

        # Check if all players have matched the bet
        for player in active_players:
            if player.current_bet < max_bet and not player.is_all_in:
                return False

        # Check that all active players have acted
        if active_player_ids - self.players_acted_this_round:
            return False

        # If there was a raise, we need to ensure we've completed a full round after the raise
        # This means all players must have acted after the raise
        if self.last_raise_player >= 0:
            # All players have acted and matched bets - round is complete
            return True

        # No raises, everyone has matched and acted - round is complete
        return True

    def _determine_winners(self) -> list[Player]:
        """Determine the winner(s) of the hand."""
        active_players = [p for p in self.players if p.is_active and p.stack >= 0]

        if len(active_players) == 0:
            return []

        if len(active_players) == 1:
            return active_players

        # Evaluate all hands
        best_rank = HandRank.HIGH_CARD
        best_tiebreakers: list[int] = []
        winners: list[Player] = []

        for player in active_players:
            all_cards = player.hand + self.community_cards
            if len(all_cards) < 5:
                continue

            rank, tiebreakers = self.hand_evaluator.evaluate_hand(all_cards)

            if rank > best_rank or (
                rank == best_rank and tiebreakers > best_tiebreakers
            ):
                best_rank = rank
                best_tiebreakers = tiebreakers
                winners = [player]
            elif rank == best_rank and tiebreakers == best_tiebreakers:
                winners.append(player)

        return winners if winners else active_players

    def _get_observation(self) -> np.ndarray:
        """Get the current observation (minimal state representation)."""
        if self.current_player_idx >= len(self.players):
            # Return zero observation if invalid state
            return np.zeros(10, dtype=np.float32)

        current_player = self.players[self.current_player_idx]

        # Player stack
        stack = float(current_player.stack)

        # Player hand evaluation (if we have enough cards)
        hand_rank = 0.0
        hand_high = 0.0
        if len(current_player.hand) == 2 and len(self.community_cards) >= 3:
            all_cards = current_player.hand + self.community_cards
            rank, tiebreakers = self.hand_evaluator.evaluate_hand(all_cards)
            hand_rank = float(rank.value)
            hand_high = float(tiebreakers[0]) if tiebreakers else 0.0
        elif len(current_player.hand) == 2:
            # Preflop or early rounds - use high card from hole cards
            hand_high = float(max(c.rank for c in current_player.hand))

        # Community cards mask (which cards are visible)
        community_mask = float(len(self.community_cards))

        # Pot size
        pot_size = float(self.pot)

        # Current bet to call
        bet_to_call = float(self._get_current_bet_to_call())

        # Position (0 = early, 1 = late)
        position = float(self.current_player_idx / max(len(self.players) - 1, 1))

        # Number of active players
        num_active = float(
            len([p for p in self.players if p.is_active and p.stack > 0])
        )

        # Is preflop
        is_preflop = 1.0 if self.betting_round == 0 else 0.0

        # First player to act in this betting round (normalized)
        # In preflop, first to act is after big blind (dealer + 3)
        # In later rounds, first to act is after dealer (dealer + 1)
        if self.betting_round == 0:
            first_player_idx = (self.dealer_idx + 3) % len(self.players)
        else:
            first_player_idx = (self.dealer_idx + 1) % len(self.players)
        # Normalize to [0, 1] range
        first_player_normalized = float(
            first_player_idx / max(len(self.players) - 1, 1)
        )

        obs = np.array(
            [
                stack,
                hand_rank,
                hand_high,
                community_mask,
                pot_size,
                bet_to_call,
                position,
                num_active,
                is_preflop,
                first_player_normalized,
            ],
            dtype=np.float32,
        )

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Get additional info about the current state."""
        return {
            "current_player": self.current_player_idx,
            "betting_round": self.betting_round,
            "pot": self.pot,
            "active_players": len(
                [p for p in self.players if p.is_active and p.stack > 0]
            ),
            "player_stacks": [p.stack for p in self.players],
        }

    def render(self) -> str | None:  # type: ignore
        """Render the current game state."""
        if self.render_mode in ("ansi", "human"):
            lines = []
            lines.append("=" * 50)
            lines.append(
                f"Betting Round: {['Preflop', 'Flop', 'Turn', 'River'][self.betting_round]}"
            )
            lines.append(f"Pot: {self.pot}")
            lines.append(f"Community Cards: {[str(c) for c in self.community_cards]}")
            lines.append("")
            for i, player in enumerate(self.players):
                marker = " <--" if i == self.current_player_idx else ""
                status = " (FOLDED)" if not player.is_active else ""
                all_in = " (ALL-IN)" if player.is_all_in else ""
                lines.append(
                    f"Player {i}: Stack={player.stack}, Bet={player.current_bet}{marker}{status}{all_in}",
                )
                if i == 0:  # Show agent's cards
                    lines.append(f"  Hand: {[str(c) for c in player.hand]}")
            lines.append("=" * 50)
            return "\n".join(lines)
        return None
