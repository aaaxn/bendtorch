import Base
import ../storage/arith.bend as Arith
import ../shape/core.bend as Shape
import ./word_injective.bend as Injective
import ../storage/model.bend as Model
import ../shape/nat_proofs.bend as NatProof

def word_has_zero(width: Nat, word: Word(width)) -> Bool:
  match width:
    case 0n:
      False{}
    case 1n+p:
      match word:
        case WCon{False{}, tail}:
          True{}
        case WCon{True{}, tail}:
          word_has_zero(p, tail)

def greater_low_bit_tail(comparison: Cmp, +bit: Bool,
  accepted: {Cmp.is_lt(Word.cmp.fin(True{}, bit, comparison)) == True{} : Bool}) ->
  {Cmp.is_lt(comparison) == True{} : Bool}:
  match comparison:
    case LT{}:
      {==}
    case EQ{}:
      match bit:
        case False{}:
          Empty.absurd({False{} == True{} : Bool}, Arith.clash(accepted))
        case True{}:
          Empty.absurd({False{} == True{} : Bool}, Arith.clash(accepted))
    case GT{}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(accepted))

def word_lt_has_zero(+width: Nat, +left: Word(width), +right: Word(width),
  accepted: {Cmp.is_lt(Word.cmp(width, left, right)) == True{} : Bool}) ->
  {word_has_zero(width, left) == True{} : Bool}:
  match width:
    case 0n:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(accepted))
    case 1n+p:
      match left right:
        case WCon{False{}, lt} WCon{rb, rt}:
          {==}
        case WCon{True{}, +lt} WCon{+rb, +rt}:
          word_lt_has_zero(p, lt, rt, greater_low_bit_tail(Word.cmp(p, lt, rt), rb, accepted))

def word_increment_exact(+width: Nat, +word: Word(width), zero: {word_has_zero(width, word) == True{} : Bool}) ->
  {Word.to_nat(width, Word.inc(width, word)) == 1n+Word.to_nat(width, word) : Nat}:
  match width:
    case 0n:
      Empty.absurd({0n == 1n : Nat}, Arith.clash(zero))
    case 1n+p:
      match word:
        case WCon{False{}, tail}:
          {==}
        case WCon{True{}, +tail}:
          Equal.cong(Nat, Nat, Nat.double, Word.to_nat(p, Word.inc(p, tail)), 1n+Word.to_nat(p, tail),
            word_increment_exact(p, tail, zero))

def increment_below_exact(+word: U32, +limit: U32, below: {U32.is_lt(word, limit) == True{} : Bool}) ->
  {U32.to_nat(U32.inc(word)) == 1n+U32.to_nat(word) : Nat}:
  match word limit:
    case U32{+w} U32{+upper}:
      word_increment_exact(32n, w, word_lt_has_zero(32n, w, upper, below))

def le_predecessor(+number: Nat, +bound: Nat, within: {Nat.is_le(1n+number, bound) == True{} : Bool}) ->
  {Nat.is_le(number, bound) == True{} : Bool}:
  match bound:
    case 0n:
      Empty.absurd({Nat.is_le(number, 0n) == True{} : Bool}, Arith.clash(within))
    case 1n+p:
      Arith.le_succ(number, p, within)

def le_successor_lt(+number: Nat, +bound: Nat, within: {Nat.is_le(1n+number, bound) == True{} : Bool}) ->
  {Nat.is_lt(number, bound) == True{} : Bool}:
  match bound:
    case 0n:
      Empty.absurd({Nat.is_lt(number, 0n) == True{} : Bool}, Arith.clash(within))
    case 1n+p:
      Arith.lt_succ(number, p, within)

def natural_word_exact(+number: Nat, +limit: U32, +within: {Nat.is_le(number, U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.trans(Nat, U32.to_nat(U32.inc(U32.from_nat(p))), 1n+U32.to_nat(U32.from_nat(p)), 1n+p,
        increment_below_exact(U32.from_nat(p), limit,
          %Arith.u32_lt(U32.from_nat(p), limit) : {_ == True{} : Bool}
          %Equal.sym(Nat, U32.to_nat(U32.from_nat(p)), p,
            natural_word_exact(p, limit, le_predecessor(p, U32.to_nat(limit), within))) :
              {Nat.is_lt(_, U32.to_nat(limit)) == True{} : Bool}
          le_successor_lt(p, U32.to_nat(limit), within)),
        Equal.cong(Nat, Nat, n => 1n+n, U32.to_nat(U32.from_nat(p)), p,
          natural_word_exact(p, limit, le_predecessor(p, U32.to_nat(limit), within))))

def le_reflexive(+number: Nat) -> {Nat.is_le(number, number) == True{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      le_reflexive(p)

def word_natural_exact(+word: U32) -> {U32.from_nat(U32.to_nat(word)) == word : U32}:
  Injective.u32_natural_injective(U32.from_nat(U32.to_nat(word)), word)(
    natural_word_exact(U32.to_nat(word), word, le_reflexive(U32.to_nat(word))))

def double_le(+left: Nat, +right: Nat, bound: {Nat.is_le(left, right) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(left), Nat.double(right)) == True{} : Bool}:
  match left right:
    case 0n 0n:
      {==}
    case 0n 1n+r:
      {==}
    case 1n+l 0n:
      Empty.absurd({Nat.is_le(2n+Nat.double(l), 0n) == True{} : Bool}, Arith.clash(bound))
    case 1n+l 1n+r:
      double_le(l, r, bound)

def le_trans(+left: Nat, +middle: Nat, +right: Nat,
  first: {Nat.is_le(left, middle) == True{} : Bool},
  second: {Nat.is_le(middle, right) == True{} : Bool}) ->
  {Nat.is_le(left, right) == True{} : Bool}:
  match left:
    case 0n:
      Arith.le_zero(right)
    case 1n+l:
      match middle:
        case 0n:
          Empty.absurd({Nat.is_le(1n+l, right) == True{} : Bool}, Arith.clash(first))
        case 1n+m:
          match right:
            case 0n:
              Empty.absurd({Nat.is_le(1n+l, 0n) == True{} : Bool}, Arith.clash(second))
            case 1n+r:
              le_trans(l, m, r, first, second)

def word_increment_le(+width: Nat, +word: Word(width)) ->
  {Nat.is_le(Word.to_nat(width, Word.inc(width, word)), 1n+Word.to_nat(width, word)) == True{} : Bool}:
  match width:
    case 0n:
      {==}
    case 1n+p:
      match word:
        case WCon{False{}, +tail}:
          le_reflexive(1n+Nat.double(Word.to_nat(p, tail)))
        case WCon{True{}, +tail}:
          double_le(Word.to_nat(p, Word.inc(p, tail)), 1n+Word.to_nat(p, tail), word_increment_le(p, tail))

def increment_le(+word: U32) ->
  {Nat.is_le(U32.to_nat(U32.inc(word)), 1n+U32.to_nat(word)) == True{} : Bool}:
  match word:
    case U32{+bits}:
      word_increment_le(32n, bits)

def natural_word_le(+number: Nat) ->
  {Nat.is_le(U32.to_nat(U32.from_nat(number)), number) == True{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      le_trans(U32.to_nat(U32.inc(U32.from_nat(p))), 1n+U32.to_nat(U32.from_nat(p)), 1n+p,
        increment_le(U32.from_nat(p)), natural_word_le(p))

def lt_le(+number: Nat, +bound: Nat, below: {Nat.is_lt(number, bound) == True{} : Bool}) ->
  {Nat.is_le(number, bound) == True{} : Bool}:
  match number bound:
    case 0n 0n:
      Empty.absurd({Nat.is_le(0n, 0n) == True{} : Bool}, Arith.clash(below))
    case 0n 1n+p:
      {==}
    case 1n+n 0n:
      Empty.absurd({Nat.is_le(1n+n, 0n) == True{} : Bool}, Arith.clash(below))
    case 1n+n 1n+p:
      lt_le(n, p, below)

def naturals_words_fitting(+dims: List<&2, U32>, +values: List<&2, Nat>,
  +fit: {True{} == Shape.fit(Shape.naturals(dims), values) : Bool}) ->
  {Shape.naturals(Shape.words(values)) == values : List<&2, Nat>}:
  match dims values:
    case Nil{} Nil{}:
      {==}
    case +dim <> +dt +head <> +tail:
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(head)), head,
        natural_word_exact(head, dim, lt_le(head, U32.to_nat(dim),
          Arith.and_left(Nat.is_lt(head, U32.to_nat(dim)), Shape.fit(Shape.naturals(dt), tail),
            Equal.sym(Bool, True{}, Shape.fit(Shape.naturals(dim <> dt), head <> tail), fit))))) :
            {_ <> Shape.naturals(Shape.words(tail)) == head <> tail : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, Shape.naturals(Shape.words(tail)), tail,
        naturals_words_fitting(dt, tail,
          Equal.sym(Bool, Shape.fit(Shape.naturals(dt), tail), True{},
            Arith.and_right(Nat.is_lt(head, U32.to_nat(dim)), Shape.fit(Shape.naturals(dt), tail),
              Equal.sym(Bool, True{}, Shape.fit(Shape.naturals(dim <> dt), head <> tail), fit))))) :
              {head <> _ == head <> tail : List<&2, Nat>}
      {==}
    case Nil{} head <> tail:
      Empty.absurd({Shape.naturals(Shape.words(head <> tail)) == head <> tail : List<&2, Nat>},
        Arith.clash(Equal.sym(Bool, True{}, False{}, fit)))
    case dim <> dt Nil{}:
      Empty.absurd({Nil{} == Nil{} : List<&2, Nat>}, Arith.clash(Equal.sym(Bool, True{}, False{}, fit)))

def add_zero(+number: Nat) -> {Nat.add(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 0n), p, add_zero(p))

def capacity_product(+left: Nat, +right: Nat) ->
  {Model.capacity(Nat.add(left, right)) == Nat.mul(Model.capacity(left), Model.capacity(right)) : Nat}:
  match left:
    case 0n:
      Equal.sym(Nat, Nat.add(Model.capacity(right), 0n), Model.capacity(right), add_zero(Model.capacity(right)))
    case 1n+p:
      Equal.trans(Nat, Nat.add(Model.capacity(Nat.add(p, right)), Model.capacity(Nat.add(p, right))),
        Nat.add(Nat.mul(Model.capacity(p), Model.capacity(right)), Nat.mul(Model.capacity(p), Model.capacity(right))),
        Nat.mul(Nat.add(Model.capacity(p), Model.capacity(p)), Model.capacity(right)),
        Equal.cong(Nat, Nat, n => Nat.add(n, n), Model.capacity(Nat.add(p, right)),
          Nat.mul(Model.capacity(p), Model.capacity(right)), capacity_product(p, right)),
        Equal.sym(Nat, Nat.mul(Nat.add(Model.capacity(p), Model.capacity(p)), Model.capacity(right)),
          Nat.add(Nat.mul(Model.capacity(p), Model.capacity(right)), Nat.mul(Model.capacity(p), Model.capacity(right))),
          NatProof.mul_add_right(Model.capacity(p), Model.capacity(p), Model.capacity(right))))

def small_power() -> {Model.capacity(8n) == 256n : Nat}:
  {==}

def chunk_power() -> {Model.capacity(12n) == 4096n : Nat}:
  {==}

def add4096(-tail: Nat) -> {Nat.add(4096n, tail) == 4096n+tail : Nat}:
  {==}

def literal_product_step(-count: Nat, -previous: Nat, -value: Nat,
  product: {Nat.mul(count, 4096n) == previous : Nat},
  chunk: {Nat.add(4096n, previous) == value : Nat}) ->
  {Nat.mul(1n+count, 4096n) == value : Nat}:
  Equal.trans(Nat, Nat.add(4096n, Nat.mul(count, 4096n)), Nat.add(4096n, previous), value,
    Equal.cong(Nat, Nat, n => Nat.add(4096n, n), Nat.mul(count, 4096n), previous, product), chunk)

def literal_product_zero() -> {Nat.mul(0n, 4096n) == 0n : Nat}:
  {==}

def literal_product_16() -> {Nat.mul(16n, 4096n) == 65536n : Nat}:
  literal_product_step(15n, 61440n, 65536n,
      literal_product_step(14n, 57344n, 61440n,
      literal_product_step(13n, 53248n, 57344n,
      literal_product_step(12n, 49152n, 53248n,
      literal_product_step(11n, 45056n, 49152n,
      literal_product_step(10n, 40960n, 45056n,
      literal_product_step(9n, 36864n, 40960n,
      literal_product_step(8n, 32768n, 36864n,
      literal_product_step(7n, 28672n, 32768n,
      literal_product_step(6n, 24576n, 28672n,
      literal_product_step(5n, 20480n, 24576n,
      literal_product_step(4n, 16384n, 20480n,
      literal_product_step(3n, 12288n, 16384n,
      literal_product_step(2n, 8192n, 12288n,
      literal_product_step(1n, 4096n, 8192n,
      literal_product_step(0n, 0n, 4096n,
      literal_product_zero(), add4096(0n)), add4096(4096n)), add4096(8192n)), add4096(12288n)), add4096(16384n)), add4096(20480n)), add4096(24576n)), add4096(28672n)), add4096(32768n)), add4096(36864n)), add4096(40960n)), add4096(45056n)), add4096(49152n)), add4096(53248n)), add4096(57344n)), add4096(61440n))

def literal_product_32() -> {Nat.mul(32n, 4096n) == 131072n : Nat}:
  literal_product_step(31n, 126976n, 131072n,
      literal_product_step(30n, 122880n, 126976n,
      literal_product_step(29n, 118784n, 122880n,
      literal_product_step(28n, 114688n, 118784n,
      literal_product_step(27n, 110592n, 114688n,
      literal_product_step(26n, 106496n, 110592n,
      literal_product_step(25n, 102400n, 106496n,
      literal_product_step(24n, 98304n, 102400n,
      literal_product_step(23n, 94208n, 98304n,
      literal_product_step(22n, 90112n, 94208n,
      literal_product_step(21n, 86016n, 90112n,
      literal_product_step(20n, 81920n, 86016n,
      literal_product_step(19n, 77824n, 81920n,
      literal_product_step(18n, 73728n, 77824n,
      literal_product_step(17n, 69632n, 73728n,
      literal_product_step(16n, 65536n, 69632n,
      literal_product_16(), add4096(65536n)), add4096(69632n)), add4096(73728n)), add4096(77824n)), add4096(81920n)), add4096(86016n)), add4096(90112n)), add4096(94208n)), add4096(98304n)), add4096(102400n)), add4096(106496n)), add4096(110592n)), add4096(114688n)), add4096(118784n)), add4096(122880n)), add4096(126976n))

def literal_product_48() -> {Nat.mul(48n, 4096n) == 196608n : Nat}:
  literal_product_step(47n, 192512n, 196608n,
      literal_product_step(46n, 188416n, 192512n,
      literal_product_step(45n, 184320n, 188416n,
      literal_product_step(44n, 180224n, 184320n,
      literal_product_step(43n, 176128n, 180224n,
      literal_product_step(42n, 172032n, 176128n,
      literal_product_step(41n, 167936n, 172032n,
      literal_product_step(40n, 163840n, 167936n,
      literal_product_step(39n, 159744n, 163840n,
      literal_product_step(38n, 155648n, 159744n,
      literal_product_step(37n, 151552n, 155648n,
      literal_product_step(36n, 147456n, 151552n,
      literal_product_step(35n, 143360n, 147456n,
      literal_product_step(34n, 139264n, 143360n,
      literal_product_step(33n, 135168n, 139264n,
      literal_product_step(32n, 131072n, 135168n,
      literal_product_32(), add4096(131072n)), add4096(135168n)), add4096(139264n)), add4096(143360n)), add4096(147456n)), add4096(151552n)), add4096(155648n)), add4096(159744n)), add4096(163840n)), add4096(167936n)), add4096(172032n)), add4096(176128n)), add4096(180224n)), add4096(184320n)), add4096(188416n)), add4096(192512n))

def literal_product_64() -> {Nat.mul(64n, 4096n) == 262144n : Nat}:
  literal_product_step(63n, 258048n, 262144n,
      literal_product_step(62n, 253952n, 258048n,
      literal_product_step(61n, 249856n, 253952n,
      literal_product_step(60n, 245760n, 249856n,
      literal_product_step(59n, 241664n, 245760n,
      literal_product_step(58n, 237568n, 241664n,
      literal_product_step(57n, 233472n, 237568n,
      literal_product_step(56n, 229376n, 233472n,
      literal_product_step(55n, 225280n, 229376n,
      literal_product_step(54n, 221184n, 225280n,
      literal_product_step(53n, 217088n, 221184n,
      literal_product_step(52n, 212992n, 217088n,
      literal_product_step(51n, 208896n, 212992n,
      literal_product_step(50n, 204800n, 208896n,
      literal_product_step(49n, 200704n, 204800n,
      literal_product_step(48n, 196608n, 200704n,
      literal_product_48(), add4096(196608n)), add4096(200704n)), add4096(204800n)), add4096(208896n)), add4096(212992n)), add4096(217088n)), add4096(221184n)), add4096(225280n)), add4096(229376n)), add4096(233472n)), add4096(237568n)), add4096(241664n)), add4096(245760n)), add4096(249856n)), add4096(253952n)), add4096(258048n))

def literal_product_80() -> {Nat.mul(80n, 4096n) == 327680n : Nat}:
  literal_product_step(79n, 323584n, 327680n,
      literal_product_step(78n, 319488n, 323584n,
      literal_product_step(77n, 315392n, 319488n,
      literal_product_step(76n, 311296n, 315392n,
      literal_product_step(75n, 307200n, 311296n,
      literal_product_step(74n, 303104n, 307200n,
      literal_product_step(73n, 299008n, 303104n,
      literal_product_step(72n, 294912n, 299008n,
      literal_product_step(71n, 290816n, 294912n,
      literal_product_step(70n, 286720n, 290816n,
      literal_product_step(69n, 282624n, 286720n,
      literal_product_step(68n, 278528n, 282624n,
      literal_product_step(67n, 274432n, 278528n,
      literal_product_step(66n, 270336n, 274432n,
      literal_product_step(65n, 266240n, 270336n,
      literal_product_step(64n, 262144n, 266240n,
      literal_product_64(), add4096(262144n)), add4096(266240n)), add4096(270336n)), add4096(274432n)), add4096(278528n)), add4096(282624n)), add4096(286720n)), add4096(290816n)), add4096(294912n)), add4096(299008n)), add4096(303104n)), add4096(307200n)), add4096(311296n)), add4096(315392n)), add4096(319488n)), add4096(323584n))

def literal_product_96() -> {Nat.mul(96n, 4096n) == 393216n : Nat}:
  literal_product_step(95n, 389120n, 393216n,
      literal_product_step(94n, 385024n, 389120n,
      literal_product_step(93n, 380928n, 385024n,
      literal_product_step(92n, 376832n, 380928n,
      literal_product_step(91n, 372736n, 376832n,
      literal_product_step(90n, 368640n, 372736n,
      literal_product_step(89n, 364544n, 368640n,
      literal_product_step(88n, 360448n, 364544n,
      literal_product_step(87n, 356352n, 360448n,
      literal_product_step(86n, 352256n, 356352n,
      literal_product_step(85n, 348160n, 352256n,
      literal_product_step(84n, 344064n, 348160n,
      literal_product_step(83n, 339968n, 344064n,
      literal_product_step(82n, 335872n, 339968n,
      literal_product_step(81n, 331776n, 335872n,
      literal_product_step(80n, 327680n, 331776n,
      literal_product_80(), add4096(327680n)), add4096(331776n)), add4096(335872n)), add4096(339968n)), add4096(344064n)), add4096(348160n)), add4096(352256n)), add4096(356352n)), add4096(360448n)), add4096(364544n)), add4096(368640n)), add4096(372736n)), add4096(376832n)), add4096(380928n)), add4096(385024n)), add4096(389120n))

def literal_product_112() -> {Nat.mul(112n, 4096n) == 458752n : Nat}:
  literal_product_step(111n, 454656n, 458752n,
      literal_product_step(110n, 450560n, 454656n,
      literal_product_step(109n, 446464n, 450560n,
      literal_product_step(108n, 442368n, 446464n,
      literal_product_step(107n, 438272n, 442368n,
      literal_product_step(106n, 434176n, 438272n,
      literal_product_step(105n, 430080n, 434176n,
      literal_product_step(104n, 425984n, 430080n,
      literal_product_step(103n, 421888n, 425984n,
      literal_product_step(102n, 417792n, 421888n,
      literal_product_step(101n, 413696n, 417792n,
      literal_product_step(100n, 409600n, 413696n,
      literal_product_step(99n, 405504n, 409600n,
      literal_product_step(98n, 401408n, 405504n,
      literal_product_step(97n, 397312n, 401408n,
      literal_product_step(96n, 393216n, 397312n,
      literal_product_96(), add4096(393216n)), add4096(397312n)), add4096(401408n)), add4096(405504n)), add4096(409600n)), add4096(413696n)), add4096(417792n)), add4096(421888n)), add4096(425984n)), add4096(430080n)), add4096(434176n)), add4096(438272n)), add4096(442368n)), add4096(446464n)), add4096(450560n)), add4096(454656n))

def literal_product_128() -> {Nat.mul(128n, 4096n) == 524288n : Nat}:
  literal_product_step(127n, 520192n, 524288n,
      literal_product_step(126n, 516096n, 520192n,
      literal_product_step(125n, 512000n, 516096n,
      literal_product_step(124n, 507904n, 512000n,
      literal_product_step(123n, 503808n, 507904n,
      literal_product_step(122n, 499712n, 503808n,
      literal_product_step(121n, 495616n, 499712n,
      literal_product_step(120n, 491520n, 495616n,
      literal_product_step(119n, 487424n, 491520n,
      literal_product_step(118n, 483328n, 487424n,
      literal_product_step(117n, 479232n, 483328n,
      literal_product_step(116n, 475136n, 479232n,
      literal_product_step(115n, 471040n, 475136n,
      literal_product_step(114n, 466944n, 471040n,
      literal_product_step(113n, 462848n, 466944n,
      literal_product_step(112n, 458752n, 462848n,
      literal_product_112(), add4096(458752n)), add4096(462848n)), add4096(466944n)), add4096(471040n)), add4096(475136n)), add4096(479232n)), add4096(483328n)), add4096(487424n)), add4096(491520n)), add4096(495616n)), add4096(499712n)), add4096(503808n)), add4096(507904n)), add4096(512000n)), add4096(516096n)), add4096(520192n))

def literal_product_144() -> {Nat.mul(144n, 4096n) == 589824n : Nat}:
  literal_product_step(143n, 585728n, 589824n,
      literal_product_step(142n, 581632n, 585728n,
      literal_product_step(141n, 577536n, 581632n,
      literal_product_step(140n, 573440n, 577536n,
      literal_product_step(139n, 569344n, 573440n,
      literal_product_step(138n, 565248n, 569344n,
      literal_product_step(137n, 561152n, 565248n,
      literal_product_step(136n, 557056n, 561152n,
      literal_product_step(135n, 552960n, 557056n,
      literal_product_step(134n, 548864n, 552960n,
      literal_product_step(133n, 544768n, 548864n,
      literal_product_step(132n, 540672n, 544768n,
      literal_product_step(131n, 536576n, 540672n,
      literal_product_step(130n, 532480n, 536576n,
      literal_product_step(129n, 528384n, 532480n,
      literal_product_step(128n, 524288n, 528384n,
      literal_product_128(), add4096(524288n)), add4096(528384n)), add4096(532480n)), add4096(536576n)), add4096(540672n)), add4096(544768n)), add4096(548864n)), add4096(552960n)), add4096(557056n)), add4096(561152n)), add4096(565248n)), add4096(569344n)), add4096(573440n)), add4096(577536n)), add4096(581632n)), add4096(585728n))

def literal_product_160() -> {Nat.mul(160n, 4096n) == 655360n : Nat}:
  literal_product_step(159n, 651264n, 655360n,
      literal_product_step(158n, 647168n, 651264n,
      literal_product_step(157n, 643072n, 647168n,
      literal_product_step(156n, 638976n, 643072n,
      literal_product_step(155n, 634880n, 638976n,
      literal_product_step(154n, 630784n, 634880n,
      literal_product_step(153n, 626688n, 630784n,
      literal_product_step(152n, 622592n, 626688n,
      literal_product_step(151n, 618496n, 622592n,
      literal_product_step(150n, 614400n, 618496n,
      literal_product_step(149n, 610304n, 614400n,
      literal_product_step(148n, 606208n, 610304n,
      literal_product_step(147n, 602112n, 606208n,
      literal_product_step(146n, 598016n, 602112n,
      literal_product_step(145n, 593920n, 598016n,
      literal_product_step(144n, 589824n, 593920n,
      literal_product_144(), add4096(589824n)), add4096(593920n)), add4096(598016n)), add4096(602112n)), add4096(606208n)), add4096(610304n)), add4096(614400n)), add4096(618496n)), add4096(622592n)), add4096(626688n)), add4096(630784n)), add4096(634880n)), add4096(638976n)), add4096(643072n)), add4096(647168n)), add4096(651264n))

def literal_product_176() -> {Nat.mul(176n, 4096n) == 720896n : Nat}:
  literal_product_step(175n, 716800n, 720896n,
      literal_product_step(174n, 712704n, 716800n,
      literal_product_step(173n, 708608n, 712704n,
      literal_product_step(172n, 704512n, 708608n,
      literal_product_step(171n, 700416n, 704512n,
      literal_product_step(170n, 696320n, 700416n,
      literal_product_step(169n, 692224n, 696320n,
      literal_product_step(168n, 688128n, 692224n,
      literal_product_step(167n, 684032n, 688128n,
      literal_product_step(166n, 679936n, 684032n,
      literal_product_step(165n, 675840n, 679936n,
      literal_product_step(164n, 671744n, 675840n,
      literal_product_step(163n, 667648n, 671744n,
      literal_product_step(162n, 663552n, 667648n,
      literal_product_step(161n, 659456n, 663552n,
      literal_product_step(160n, 655360n, 659456n,
      literal_product_160(), add4096(655360n)), add4096(659456n)), add4096(663552n)), add4096(667648n)), add4096(671744n)), add4096(675840n)), add4096(679936n)), add4096(684032n)), add4096(688128n)), add4096(692224n)), add4096(696320n)), add4096(700416n)), add4096(704512n)), add4096(708608n)), add4096(712704n)), add4096(716800n))

def literal_product_192() -> {Nat.mul(192n, 4096n) == 786432n : Nat}:
  literal_product_step(191n, 782336n, 786432n,
      literal_product_step(190n, 778240n, 782336n,
      literal_product_step(189n, 774144n, 778240n,
      literal_product_step(188n, 770048n, 774144n,
      literal_product_step(187n, 765952n, 770048n,
      literal_product_step(186n, 761856n, 765952n,
      literal_product_step(185n, 757760n, 761856n,
      literal_product_step(184n, 753664n, 757760n,
      literal_product_step(183n, 749568n, 753664n,
      literal_product_step(182n, 745472n, 749568n,
      literal_product_step(181n, 741376n, 745472n,
      literal_product_step(180n, 737280n, 741376n,
      literal_product_step(179n, 733184n, 737280n,
      literal_product_step(178n, 729088n, 733184n,
      literal_product_step(177n, 724992n, 729088n,
      literal_product_step(176n, 720896n, 724992n,
      literal_product_176(), add4096(720896n)), add4096(724992n)), add4096(729088n)), add4096(733184n)), add4096(737280n)), add4096(741376n)), add4096(745472n)), add4096(749568n)), add4096(753664n)), add4096(757760n)), add4096(761856n)), add4096(765952n)), add4096(770048n)), add4096(774144n)), add4096(778240n)), add4096(782336n))

def literal_product_208() -> {Nat.mul(208n, 4096n) == 851968n : Nat}:
  literal_product_step(207n, 847872n, 851968n,
      literal_product_step(206n, 843776n, 847872n,
      literal_product_step(205n, 839680n, 843776n,
      literal_product_step(204n, 835584n, 839680n,
      literal_product_step(203n, 831488n, 835584n,
      literal_product_step(202n, 827392n, 831488n,
      literal_product_step(201n, 823296n, 827392n,
      literal_product_step(200n, 819200n, 823296n,
      literal_product_step(199n, 815104n, 819200n,
      literal_product_step(198n, 811008n, 815104n,
      literal_product_step(197n, 806912n, 811008n,
      literal_product_step(196n, 802816n, 806912n,
      literal_product_step(195n, 798720n, 802816n,
      literal_product_step(194n, 794624n, 798720n,
      literal_product_step(193n, 790528n, 794624n,
      literal_product_step(192n, 786432n, 790528n,
      literal_product_192(), add4096(786432n)), add4096(790528n)), add4096(794624n)), add4096(798720n)), add4096(802816n)), add4096(806912n)), add4096(811008n)), add4096(815104n)), add4096(819200n)), add4096(823296n)), add4096(827392n)), add4096(831488n)), add4096(835584n)), add4096(839680n)), add4096(843776n)), add4096(847872n))

def literal_product_224() -> {Nat.mul(224n, 4096n) == 917504n : Nat}:
  literal_product_step(223n, 913408n, 917504n,
      literal_product_step(222n, 909312n, 913408n,
      literal_product_step(221n, 905216n, 909312n,
      literal_product_step(220n, 901120n, 905216n,
      literal_product_step(219n, 897024n, 901120n,
      literal_product_step(218n, 892928n, 897024n,
      literal_product_step(217n, 888832n, 892928n,
      literal_product_step(216n, 884736n, 888832n,
      literal_product_step(215n, 880640n, 884736n,
      literal_product_step(214n, 876544n, 880640n,
      literal_product_step(213n, 872448n, 876544n,
      literal_product_step(212n, 868352n, 872448n,
      literal_product_step(211n, 864256n, 868352n,
      literal_product_step(210n, 860160n, 864256n,
      literal_product_step(209n, 856064n, 860160n,
      literal_product_step(208n, 851968n, 856064n,
      literal_product_208(), add4096(851968n)), add4096(856064n)), add4096(860160n)), add4096(864256n)), add4096(868352n)), add4096(872448n)), add4096(876544n)), add4096(880640n)), add4096(884736n)), add4096(888832n)), add4096(892928n)), add4096(897024n)), add4096(901120n)), add4096(905216n)), add4096(909312n)), add4096(913408n))

def literal_product_240() -> {Nat.mul(240n, 4096n) == 983040n : Nat}:
  literal_product_step(239n, 978944n, 983040n,
      literal_product_step(238n, 974848n, 978944n,
      literal_product_step(237n, 970752n, 974848n,
      literal_product_step(236n, 966656n, 970752n,
      literal_product_step(235n, 962560n, 966656n,
      literal_product_step(234n, 958464n, 962560n,
      literal_product_step(233n, 954368n, 958464n,
      literal_product_step(232n, 950272n, 954368n,
      literal_product_step(231n, 946176n, 950272n,
      literal_product_step(230n, 942080n, 946176n,
      literal_product_step(229n, 937984n, 942080n,
      literal_product_step(228n, 933888n, 937984n,
      literal_product_step(227n, 929792n, 933888n,
      literal_product_step(226n, 925696n, 929792n,
      literal_product_step(225n, 921600n, 925696n,
      literal_product_step(224n, 917504n, 921600n,
      literal_product_224(), add4096(917504n)), add4096(921600n)), add4096(925696n)), add4096(929792n)), add4096(933888n)), add4096(937984n)), add4096(942080n)), add4096(946176n)), add4096(950272n)), add4096(954368n)), add4096(958464n)), add4096(962560n)), add4096(966656n)), add4096(970752n)), add4096(974848n)), add4096(978944n))

def literal_product_256() -> {Nat.mul(256n, 4096n) == 1048576n : Nat}:
  literal_product_step(255n, 1044480n, 1048576n,
      literal_product_step(254n, 1040384n, 1044480n,
      literal_product_step(253n, 1036288n, 1040384n,
      literal_product_step(252n, 1032192n, 1036288n,
      literal_product_step(251n, 1028096n, 1032192n,
      literal_product_step(250n, 1024000n, 1028096n,
      literal_product_step(249n, 1019904n, 1024000n,
      literal_product_step(248n, 1015808n, 1019904n,
      literal_product_step(247n, 1011712n, 1015808n,
      literal_product_step(246n, 1007616n, 1011712n,
      literal_product_step(245n, 1003520n, 1007616n,
      literal_product_step(244n, 999424n, 1003520n,
      literal_product_step(243n, 995328n, 999424n,
      literal_product_step(242n, 991232n, 995328n,
      literal_product_step(241n, 987136n, 991232n,
      literal_product_step(240n, 983040n, 987136n,
      literal_product_240(), add4096(983040n)), add4096(987136n)), add4096(991232n)), add4096(995328n)), add4096(999424n)), add4096(1003520n)), add4096(1007616n)), add4096(1011712n)), add4096(1015808n)), add4096(1019904n)), add4096(1024000n)), add4096(1028096n)), add4096(1032192n)), add4096(1036288n)), add4096(1040384n)), add4096(1044480n))

def capacity_word_bits() -> {1048576 == U32{Arith.unit(32n, 20n)} : U32}:
  {==}

def u32_unit_nat(+bit: Nat, below: {Nat.is_lt(bit, 32n) == True{} : Bool}) ->
  {Model.capacity(bit) == U32.to_nat(U32{Arith.unit(32n, bit)}) : Nat}:
  Arith.unit_nat(32n, bit, below)

def capacity_word_model() -> {U32.to_nat(1048576) == Model.capacity(20n) : Nat}:
  Equal.trans(Nat, U32.to_nat(1048576), U32.to_nat(U32{Arith.unit(32n, 20n)}), Model.capacity(20n),
    Equal.cong(U32, Nat, U32.to_nat, 1048576, U32{Arith.unit(32n, 20n)}, capacity_word_bits()),
    Equal.sym(Nat, Model.capacity(20n), U32.to_nat(U32{Arith.unit(32n, 20n)}), u32_unit_nat(20n, {==})))

def capacity_product_at(+left: Nat, +right: Nat, +sum: Nat, equal: {Nat.add(left, right) == sum : Nat}) ->
  {Model.capacity(sum) == Nat.mul(Model.capacity(left), Model.capacity(right)) : Nat}:
  Equal.trans(Nat, Model.capacity(sum), Model.capacity(Nat.add(left, right)), Nat.mul(Model.capacity(left), Model.capacity(right)),
    Equal.sym(Nat, Model.capacity(Nat.add(left, right)), Model.capacity(sum),
      Equal.cong(Nat, Nat, Model.capacity, Nat.add(left, right), sum, equal)),
    capacity_product(left, right))

def capacity_model_product() -> {Model.capacity(20n) == Nat.mul(256n, 4096n) : Nat}:
  Equal.trans(Nat, Model.capacity(20n), Nat.mul(Model.capacity(8n), Model.capacity(12n)), Nat.mul(256n, 4096n),
    capacity_product_at(8n, 12n, 20n, {==}),
    Equal.trans(Nat, Nat.mul(Model.capacity(8n), Model.capacity(12n)), Nat.mul(256n, Model.capacity(12n)), Nat.mul(256n, 4096n),
      Equal.cong(Nat, Nat, n => Nat.mul(n, Model.capacity(12n)), Model.capacity(8n), 256n, small_power()),
      Equal.cong(Nat, Nat, n => Nat.mul(256n, n), Model.capacity(12n), 4096n, chunk_power())))

def capacity_word_nat() -> {U32.to_nat(1048576) == 1048576n : Nat}:
  Equal.trans(Nat, U32.to_nat(1048576), Model.capacity(20n), 1048576n, capacity_word_model(),
    Equal.trans(Nat, Model.capacity(20n), Nat.mul(256n, 4096n), 1048576n,
      capacity_model_product(), literal_product_256()))

def capacity_within(+number: Nat, within: Shape.LE(number, 1048576n)) ->
  {Nat.is_le(number, U32.to_nat(1048576)) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_le(number, U32.to_nat(1048576)), Nat.is_le(number, 1048576n), True{},
    Equal.cong(Nat, Bool, n => Nat.is_le(number, n), U32.to_nat(1048576), 1048576n, capacity_word_nat()),
    Equal.sym(Bool, True{}, Nat.is_le(number, 1048576n), NatProof.le_true(number, 1048576n, within)))

def capacity_natural_word_exact(+number: Nat, within: Shape.LE(number, 1048576n)) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}:
  natural_word_exact(number, 1048576, capacity_within(number, within))

def shape_capacity(+shape: Shape.Shape) -> Shape.LE(Shape.elements(shape), 1048576n):
  match shape:
    case Shape.SShape{+dims, +size, +exact, rank_bound, capacity_bound}:
      %Equal.sym(Nat, size, Shape.logical_numel(dims), exact) : Shape.LE(_, 1048576n)
      capacity_bound

def shape_count_exact(+shape: Shape.Shape) -> {U32.to_nat(Shape.count(shape)) == Shape.elements(shape) : Nat}:
  capacity_natural_word_exact(Shape.elements(shape), shape_capacity(shape))

def shape_count_within(+shape: Shape.Shape) -> {U32.is_le(Shape.count(shape), 1048576) == True{} : Bool}:
  %Arith.u32_le(Shape.count(shape), 1048576) : {_ == True{} : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), shape_count_exact(shape)) :
    {Nat.is_le(_, U32.to_nat(1048576)) == True{} : Bool}
  capacity_within(Shape.elements(shape), shape_capacity(shape))
