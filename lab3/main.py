import math
import random
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Iterable, Set

import matplotlib.pyplot as plt

EPS = 1e-9


# =========================================================
# БАЗОВЫЕ СТРУКТУРЫ
# =========================================================

@dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    name: str = ""

    def is_vertical(self) -> bool:
        return abs(self.a.x - self.b.x) < EPS

    def left_right(self) -> Tuple[Point, Point]:
        if (self.a.x, self.a.y) <= (self.b.x, self.b.y):
            return self.a, self.b
        return self.b, self.a

    def as_vector(self) -> Tuple[float, float]:
        return self.b.x - self.a.x, self.b.y - self.a.y


# =========================================================
# БАЗОВАЯ ГЕОМЕТРИЯ
# =========================================================

def almost_equal(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def point_key(p: Point, digits: int = 8) -> Tuple[float, float]:
    return round(p.x, digits), round(p.y, digits)


def segment_name_pair_key(s1: Segment, s2: Segment) -> Tuple[str, str]:
    return tuple(sorted((s1.name, s2.name)))


def seg_key(seg: Segment, digits: int = 8):
    p1 = point_key(seg.a, digits)
    p2 = point_key(seg.b, digits)
    if p2 < p1:
        p1, p2 = p2, p1
    return p1, p2


def cross_vec(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def cross(o: Point, a: Point, b: Point) -> float:
    return cross_vec(a.x - o.x, a.y - o.y, b.x - o.x, b.y - o.y)


def orientation(a: Point, b: Point, c: Point) -> int:
    val = cross(a, b, c)
    if val > EPS:
        return 1
    if val < -EPS:
        return -1
    return 0


def point_in_bbox(p: Point, a: Point, b: Point) -> bool:
    return (
        min(a.x, b.x) - EPS <= p.x <= max(a.x, b.x) + EPS
        and min(a.y, b.y) - EPS <= p.y <= max(a.y, b.y) + EPS
    )


def point_on_segment(p: Point, s: Segment) -> bool:
    return orientation(s.a, s.b, p) == 0 and point_in_bbox(p, s.a, s.b)


def line_coefficients(s: Segment) -> Tuple[float, float, float]:
    """
    Прямая через две точки:
    A*x + B*y + C = 0
    """
    A = s.a.y - s.b.y
    B = s.b.x - s.a.x
    C = s.a.x * s.b.y - s.b.x * s.a.y
    return A, B, C


def line_intersection_point(s1: Segment, s2: Segment) -> Optional[Point]:
    A1, B1, C1 = line_coefficients(s1)
    A2, B2, C2 = line_coefficients(s2)

    det = A1 * B2 - A2 * B1
    if abs(det) < EPS:
        return None

    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return Point(x, y)


def _project_overlap_1d(a1: float, a2: float, b1: float, b2: float) -> Optional[Tuple[float, float]]:
    left = max(min(a1, a2), min(b1, b2))
    right = min(max(a1, a2), max(b1, b2))
    if left > right + EPS:
        return None
    return left, right


def _point_from_parameter(seg: Segment, t: float) -> Point:
    return Point(seg.a.x + t * (seg.b.x - seg.a.x), seg.a.y + t * (seg.b.y - seg.a.y))


def segment_intersection_detail(s1: Segment, s2: Segment) -> Dict[str, Any]:
    """
    Возвращает:
    {"type": "none"}
    {"type": "point", "point": Point}
    {"type": "overlap", "segment": Segment}
    """
    p = s1.a
    r = Point(s1.b.x - s1.a.x, s1.b.y - s1.a.y)
    q = s2.a
    s = Point(s2.b.x - s2.a.x, s2.b.y - s2.a.y)

    rxs = cross_vec(r.x, r.y, s.x, s.y)
    qmp = Point(q.x - p.x, q.y - p.y)
    qmpxr = cross_vec(qmp.x, qmp.y, r.x, r.y)

    # Коллинеарный случай
    if abs(rxs) < EPS and abs(qmpxr) < EPS:
        rr = r.x * r.x + r.y * r.y
        if rr < EPS:
            if point_on_segment(p, s2):
                return {"type": "point", "point": p}
            return {"type": "none"}

        t0 = ((q.x - p.x) * r.x + (q.y - p.y) * r.y) / rr
        t1 = t0 + (s.x * r.x + s.y * r.y) / rr
        lo = max(0.0, min(t0, t1))
        hi = min(1.0, max(t0, t1))

        if lo > hi + EPS:
            return {"type": "none"}

        p1 = _point_from_parameter(s1, lo)
        p2 = _point_from_parameter(s1, hi)

        if abs(p1.x - p2.x) < EPS and abs(p1.y - p2.y) < EPS:
            return {"type": "point", "point": p1}

        if (p2.x, p2.y) < (p1.x, p1.y):
            p1, p2 = p2, p1
        return {"type": "overlap", "segment": Segment(p1, p2, f"{s1.name}&{s2.name}")}

    # Параллельные непересекающиеся
    if abs(rxs) < EPS:
        return {"type": "none"}

    t = cross_vec(qmp.x, qmp.y, s.x, s.y) / rxs
    u = cross_vec(qmp.x, qmp.y, r.x, r.y) / rxs

    if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
        inter = _point_from_parameter(s1, t)
        return {"type": "point", "point": inter}

    return {"type": "none"}


# =========================================================
# ЗАДАНИЕ 1a. ПО УРАВНЕНИЯМ ПРЯМЫХ
# =========================================================

def intersections_by_line_equations(segments: List[Segment], skip_collinear: bool = True):
    """
    По уравнениям прямых:
    1) строим прямые, содержащие отрезки,
    2) находим точку пересечения прямых,
    3) проверяем, лежит ли она на обоих отрезках.

    Для коллинеарных наложений:
    - если skip_collinear=True, пропускаем;
    - если False, возвращаем overlap через общий детальный метод.
    """
    results = []

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]

            A1, B1, C1 = line_coefficients(s1)
            A2, B2, C2 = line_coefficients(s2)

            det = A1 * B2 - A2 * B1

            # Прямые параллельны
            if abs(det) < EPS:
                # Проверим, совпадают ли прямые
                if (
                    abs(A1 * B2 - A2 * B1) < EPS and
                    abs(A1 * C2 - A2 * C1) < EPS and
                    abs(B1 * C2 - B2 * C1) < EPS
                ):
                    detail = segment_intersection_detail(s1, s2)
                    if detail["type"] == "point":
                        results.append({
                            "segments": (s1.name, s2.name),
                            "type": "point",
                            "point": detail["point"],
                            "segment": None
                        })
                    elif detail["type"] == "overlap" and not skip_collinear:
                        results.append({
                            "segments": (s1.name, s2.name),
                            "type": "overlap",
                            "point": None,
                            "segment": detail["segment"]
                        })
                continue

            # Точка пересечения бесконечных прямых
            p = line_intersection_point(s1, s2)
            if p is None:
                continue

            # Проверка принадлежности обоим отрезкам
            if point_on_segment(p, s1) and point_on_segment(p, s2):
                results.append({
                    "segments": (s1.name, s2.name),
                    "type": "point",
                    "point": p,
                    "segment": None
                })

    return results


# =========================================================
# ЗАДАНИЕ 1b. МЕТОД КОСЫХ ПРОИЗВЕДЕНИЙ
# =========================================================

def segments_intersect_cross_method(s1: Segment, s2: Segment, allow_collinear: bool = False) -> bool:
    o1 = orientation(s1.a, s1.b, s2.a)
    o2 = orientation(s1.a, s1.b, s2.b)
    o3 = orientation(s2.a, s2.b, s1.a)
    o4 = orientation(s2.a, s2.b, s1.b)

    if o1 != o2 and o3 != o4:
        return True

    if allow_collinear:
        return segment_intersection_detail(s1, s2)["type"] != "none"

    return False


def intersections_by_cross_products(segments: List[Segment], allow_collinear: bool = False):
    """
    Метод косых произведений:
    - определяет факт пересечения;
    - НЕ вычисляет координаты точки пересечения для неколлинеарного случая.
    """
    results = []

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]

            if not segments_intersect_cross_method(s1, s2, allow_collinear=allow_collinear):
                continue

            detail = segment_intersection_detail(s1, s2)

            if detail["type"] == "none":
                continue

            if detail["type"] == "overlap":
                if allow_collinear:
                    results.append({
                        "segments": (s1.name, s2.name),
                        "type": "overlap",
                        "point": None,
                        "segment": detail["segment"]
                    })
            else:
                results.append({
                    "segments": (s1.name, s2.name),
                    "type": "intersect",
                    "point": None,
                    "segment": None
                })

    return results


# =========================================================
# ЗАДАНИЕ 1c. ЗАМЕТАЮЩАЯ ПРЯМАЯ
# =========================================================

def y_at(seg: Segment, x: float) -> float:
    if abs(seg.a.x - seg.b.x) < EPS:
        return min(seg.a.y, seg.b.y)
    t = (x - seg.a.x) / (seg.b.x - seg.a.x)
    return seg.a.y + t * (seg.b.y - seg.a.y)


def _intersection_event_point(s1: Segment, s2: Segment) -> Optional[Point]:
    detail = segment_intersection_detail(s1, s2)
    if detail["type"] == "point":
        return detail["point"]
    return None


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def sweep_line_intersections(segments: List[Segment], allow_collinear: bool = False):
    """
    Корректная учебная реализация sweep line.

    Для неколлинеарного случая:
      - в очередь событий кладутся левые/правые концы,
      - при обнаружении пересечения соседей оно добавляется как событие,
      - в точке пересечения порядок активных отрезков меняется.

    Для коллинеарных наложений:
      - они дополнительно обнаруживаются прямой проверкой пар и
        добавляются в результат как overlap.
      - это стандартная "доработка для вырожденных случаев" поверх
        базового Bentley–Ottmann для общего положения.
    """
    for s in segments:
        if s.is_vertical():
            raise ValueError(f"Отрезок {s.name} вертикальный, это запрещено по условию.")

    seg_by_name = {s.name: s for s in segments}

    # priority: 0 = intersection, 1 = start, 2 = end
    event_heap: List[Tuple[float, float, int, Tuple[str, ...]]] = []
    for s in segments:
        left, right = s.left_right()
        heapq.heappush(event_heap, (left.x, left.y, 1, (s.name,)))
        heapq.heappush(event_heap, (right.x, right.y, 2, (s.name,)))

    active: List[str] = []
    results: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    scheduled_intersections: Set[Tuple[float, float, str, str]] = set()
    current_x = -10.0**18

    def active_sort_key(name: str):
        seg = seg_by_name[name]
        x_probe = current_x + 1e-7
        return (y_at(seg, x_probe), name)

    def sort_active():
        active.sort(key=active_sort_key)

    def find_pos(name: str) -> int:
        return active.index(name)

    def report_pair(s1: Segment, s2: Segment):
        pair_key = _normalize_pair(s1.name, s2.name)
        if pair_key in seen_pairs:
            return
        detail = segment_intersection_detail(s1, s2)
        if detail["type"] == "none":
            return
        if detail["type"] == "overlap" and not allow_collinear:
            return
        seen_pairs.add(pair_key)
        results.append({
            "segments": pair_key,
            "type": detail["type"],
            "point": detail.get("point"),
            "segment": detail.get("segment"),
        })

    def schedule_if_needed(name1: Optional[str], name2: Optional[str]):
        if name1 is None or name2 is None or name1 == name2:
            return
        s1 = seg_by_name[name1]
        s2 = seg_by_name[name2]
        detail = segment_intersection_detail(s1, s2)

        if detail["type"] == "none":
            return

        if detail["type"] == "overlap":
            if allow_collinear:
                report_pair(s1, s2)
            return

        p = detail["point"]
        # Для будущих событий нужны только пересечения правее sweep line,
        # либо в текущем x, но выше текущего события.
        if p.x < current_x - EPS:
            return

        key = (round(p.x, 8), round(p.y, 8), *_normalize_pair(name1, name2))
        if key not in scheduled_intersections:
            scheduled_intersections.add(key)
            heapq.heappush(event_heap, (p.x, p.y, 0, _normalize_pair(name1, name2)))

    while event_heap:
        x, y, typ, names = heapq.heappop(event_heap)
        current_x = x

        # сгруппировать все события в одной точке
        batch = [(x, y, typ, names)]
        while event_heap and abs(event_heap[0][0] - x) < EPS and abs(event_heap[0][1] - y) < EPS:
            batch.append(heapq.heappop(event_heap))

        point = Point(x, y)

        start_names = []
        end_names = []
        intersection_pairs = []

        for _, _, etyp, payload in batch:
            if etyp == 1:
                start_names.extend(payload)
            elif etyp == 2:
                end_names.extend(payload)
            else:
                intersection_pairs.append(payload)

        # Все сегменты, проходящие через точку события
        crossing_now = set(start_names + end_names)
        for a, b in intersection_pairs:
            crossing_now.add(a)
            crossing_now.add(b)

        # Для надёжности добавим активные отрезки, которые проходят через эту точку.
        for name in active:
            if point_on_segment(point, seg_by_name[name]):
                crossing_now.add(name)

        crossing_now = list(crossing_now)

        # Сначала отчитаем все пары сегментов, реально проходящие через точку
        for i in range(len(crossing_now)):
            for j in range(i + 1, len(crossing_now)):
                s1 = seg_by_name[crossing_now[i]]
                s2 = seg_by_name[crossing_now[j]]
                detail = segment_intersection_detail(s1, s2)
                if detail["type"] == "point":
                    p = detail["point"]
                    if abs(p.x - x) < 1e-7 and abs(p.y - y) < 1e-7:
                        report_pair(s1, s2)
                elif detail["type"] == "overlap" and allow_collinear:
                    report_pair(s1, s2)

        # Удалить сегменты, заканчивающиеся или проходящие через intersection-event.
        to_remove = set(end_names)
        for a, b in intersection_pairs:
            to_remove.add(a)
            to_remove.add(b)

        active = [name for name in active if name not in to_remove]
        sort_active()

        # Добавить сегменты, начинающиеся или проходящие через intersection-event.
        to_add = set(start_names)
        for a, b in intersection_pairs:
            to_add.add(a)
            to_add.add(b)

        for name in to_add:
            if name not in active:
                active.append(name)
        sort_active()

        # Проверить соседей всех сегментов, затронутых событием.
        touched = set(crossing_now)
        touched.update(start_names)
        touched.update(end_names)

        neighbor_candidates = set()
        sort_active()
        for name in touched:
            if name in active:
                pos = find_pos(name)
                if pos - 1 >= 0:
                    neighbor_candidates.add(_normalize_pair(active[pos - 1], name))
                if pos + 1 < len(active):
                    neighbor_candidates.add(_normalize_pair(name, active[pos + 1]))

        # Также после удаления/добавления могли стать соседями бывшие "краевые" сегменты.
        sort_active()
        for idx in range(len(active) - 1):
            a = active[idx]
            b = active[idx + 1]
            # Достаточно локально проверять пары рядом с затронутыми,
            # но для учебной устойчивости добавим только если хотя бы один затронут.
            if a in touched or b in touched:
                neighbor_candidates.add(_normalize_pair(a, b))

        for a, b in sorted(neighbor_candidates):
            schedule_if_needed(a, b)

    # Для части II нужно обязательно добавить все коллинеарные наложения.
    if allow_collinear:
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                detail = segment_intersection_detail(segments[i], segments[j])
                if detail["type"] == "overlap":
                    report_pair(segments[i], segments[j])

    return results


# =========================================================
# ГЕНЕРАЦИЯ ОТРЕЗКОВ ПОД ЛР3
# =========================================================

def random_point(xmin=0, xmax=40, ymin=0, ymax=40) -> Point:
    return Point(random.randint(xmin, xmax), random.randint(ymin, ymax))


def random_non_vertical_segment(name: str,
                                xmin=0, xmax=40,
                                ymin=0, ymax=40,
                                max_attempts: int = 1000) -> Segment:
    for _ in range(max_attempts):
        p1 = random_point(xmin, xmax, ymin, ymax)
        p2 = random_point(xmin, xmax, ymin, ymax)

        if p1 == p2:
            continue
        if abs(p1.x - p2.x) < EPS:
            continue

        return Segment(p1, p2, name)

    raise ValueError("Не удалось сгенерировать не вертикальный отрезок.")


def is_duplicate_segment(new_seg: Segment, segments: List[Segment]) -> bool:
    for s in segments:
        same_dir = (
            abs(new_seg.a.x - s.a.x) < EPS and abs(new_seg.a.y - s.a.y) < EPS and
            abs(new_seg.b.x - s.b.x) < EPS and abs(new_seg.b.y - s.b.y) < EPS
        )
        rev_dir = (
            abs(new_seg.a.x - s.b.x) < EPS and abs(new_seg.a.y - s.b.y) < EPS and
            abs(new_seg.b.x - s.a.x) < EPS and abs(new_seg.b.y - s.a.y) < EPS
        )
        if same_dir or rev_dir:
            return True
    return False


def count_noncollinear_intersections(segments: List[Segment]) -> int:
    cnt = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            detail = segment_intersection_detail(segments[i], segments[j])
            if detail["type"] == "point":
                o1 = orientation(segments[i].a, segments[i].b, segments[j].a)
                o2 = orientation(segments[i].a, segments[i].b, segments[j].b)
                o3 = orientation(segments[j].a, segments[j].b, segments[i].a)
                o4 = orientation(segments[j].a, segments[j].b, segments[i].b)
                if not (o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0):
                    cnt += 1
    return cnt


def generate_random_segments_for_lr3(n: int,
                                     xmin=0, xmax=40,
                                     ymin=0, ymax=40,
                                     seed: Optional[int] = None) -> List[Segment]:
    """
    Генерация:
    - n >= 10
    - нет вертикальных
    - >= 2 пары пересекающихся неколлинеарных
    - 1 коллинеарная пара с наложением
    - 1 коллинеарная пара без наложения
    """
    if n < 10:
        raise ValueError("По условию ЛР3 n должно быть не меньше 10.")

    if seed is not None:
        random.seed(seed)

    segments = []

    fixed = [
        Segment(Point(2, 2), Point(12, 12), "s1"),
        Segment(Point(2, 12), Point(12, 2), "s2"),

        Segment(Point(15, 5), Point(25, 15), "s3"),
        Segment(Point(15, 15), Point(25, 5), "s4"),

        # коллинеарные с наложением
        Segment(Point(5, 25), Point(15, 25), "s5"),
        Segment(Point(10, 25), Point(20, 25), "s6"),

        # коллинеарные без наложения
        Segment(Point(5, 30), Point(10, 30), "s7"),
        Segment(Point(15, 30), Point(20, 30), "s8"),
    ]
    segments.extend(fixed)

    idx = 9
    while len(segments) < n:
        seg = random_non_vertical_segment(f"s{idx}", xmin, xmax, ymin, ymax)
        if not is_duplicate_segment(seg, segments):
            segments.append(seg)
            idx += 1

    for s in segments:
        if s.is_vertical():
            raise ValueError("Сгенерирован вертикальный отрезок, чего быть не должно.")

    if count_noncollinear_intersections(segments[:4]) < 2:
        raise ValueError("Не удалось гарантировать две пары пересекающихся отрезков.")

    return segments


def split_segments_for_task(segments: List[Segment]) -> Tuple[List[Segment], List[Segment]]:
    """
    Для пункта I берём только неколлинеарные.
    Для пункта II добавляем коллинеарные.
    """
    non_collinear = []
    with_collinear = segments[:]

    for s in segments:
        if s.name in ("s5", "s6", "s7", "s8"):
            continue
        non_collinear.append(s)

    return non_collinear, with_collinear


# =========================================================
# ЛОКАЛИЗАЦИЯ ТОЧКИ В МНОГОУГОЛЬНИКЕ
# =========================================================

def polygon_edges(poly: List[Point]) -> List[Segment]:
    return [Segment(poly[i], poly[(i + 1) % len(poly)], f"e{i+1}") for i in range(len(poly))]


def point_on_polygon_boundary(p: Point, poly: List[Point]) -> bool:
    return any(point_on_segment(p, e) for e in polygon_edges(poly))


def signed_angle(p: Point, a: Point, b: Point) -> float:
    ux, uy = a.x - p.x, a.y - p.y
    vx, vy = b.x - p.x, b.y - p.y

    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu < EPS or nv < EPS:
        return 0.0

    cr = ux * vy - uy * vx
    dt = ux * vx + uy * vy
    return math.atan2(cr, dt)


def locate_point_angle_method(p: Point, poly: List[Point]) -> str:
    if point_on_polygon_boundary(p, poly):
        return "boundary"

    total = 0.0
    n = len(poly)
    for i in range(n):
        total += signed_angle(p, poly[i], poly[(i + 1) % n])

    return "inside" if abs(total) > math.pi else "outside"


def locate_point_ray_method(p: Point, poly: List[Point]) -> str:
    """
    Горизонтальный луч вправо.
    Граница проверяется отдельно.
    Далее используем стандартное полуоткрытое правило:
    ребро учитывается, если уровни по y "строго по разные стороны" от точки.
    Это устраняет двойной счёт в вершинах.
    """
    if point_on_polygon_boundary(p, poly):
        return "boundary"

    inside = False
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if (a.y > p.y) != (b.y > p.y):
            x_inter = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if x_inter > p.x + EPS:
                inside = not inside

    return "inside" if inside else "outside"


# =========================================================
# ГЕНЕРАЦИЯ МНОГОУГОЛЬНИКА И ТОЧЕК
# =========================================================

def build_nonconvex_polygon() -> List[Point]:
    return [
        Point(0, 0),
        Point(4, 0),
        Point(6, 2),
        Point(8, 0),
        Point(12, 0),
        Point(12, 6),
        Point(9, 6),
        Point(7, 3),   # вогнутая вершина
        Point(5, 8),
        Point(0, 6),
    ]


def build_modified_polygon_with_horizontal_vertices() -> List[Point]:
    """
    Не менее трёх вершин на одной горизонтали y = 6,
    причём две соседние.
    """
    return [
        Point(0, 0),
        Point(4, 0),
        Point(6, 2),
        Point(8, 0),
        Point(12, 0),
        Point(12, 6),
        Point(9, 6),   # соседняя на той же горизонтали
        Point(7, 3),
        Point(5, 6),   # третья на горизонтали
        Point(0, 6),
    ]


def generate_random_points_for_polygon(k: int,
                                       xmin=-2, xmax=14,
                                       ymin=-2, ymax=10,
                                       seed: Optional[int] = None) -> List[Point]:
    if k < 10:
        raise ValueError("По условию ЛР3 k должно быть не меньше 10.")

    if seed is not None:
        random.seed(seed + 1000)

    pts = []
    for _ in range(k):
        pts.append(Point(random.randint(xmin, xmax), random.randint(ymin, ymax)))
    return pts


# =========================================================
# ВЫВОД И ВИЗУАЛИЗАЦИЯ
# =========================================================

def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(item):
        if item["type"] == "point":
            p = item["point"]
            return (0, round(p.x, 8), round(p.y, 8), item["segments"])

        if item["type"] == "intersect":
            return (1, item["segments"])

        seg = item["segment"]
        return (
            2,
            round(seg.a.x, 8), round(seg.a.y, 8),
            round(seg.b.x, 8), round(seg.b.y, 8),
            item["segments"]
        )

    return sorted(results, key=key)


def print_segments_list(title: str, segments: List[Segment]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for s in segments:
        print(f"{s.name}: ({s.a.x}, {s.a.y}) -> ({s.b.x}, {s.b.y})")


def print_intersections(title: str, results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    results = sort_results(results)

    if not results:
        print("Пересечений не найдено.")
        return

    for i, item in enumerate(results, 1):
        s1, s2 = item["segments"]

        if item["type"] == "point":
            p = item["point"]
            print(f"{i:2d}. {s1} и {s2} -> точка ({p.x:.6f}, {p.y:.6f})")

        elif item["type"] == "intersect":
            print(f"{i:2d}. {s1} и {s2} -> пересекаются")

        elif item["type"] == "overlap":
            seg = item["segment"]
            print(
                f"{i:2d}. {s1} и {s2} -> наложение "
                f"[({seg.a.x:.6f}, {seg.a.y:.6f}) - ({seg.b.x:.6f}, {seg.b.y:.6f})]"
            )


def print_point_locations(title: str, poly: List[Point], points: List[Point]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("Угловой метод:")
    for i, p in enumerate(points, 1):
        print(f"M{i:02d} = ({p.x:.2f}, {p.y:.2f}) -> {locate_point_angle_method(p, poly)}")

    print("\nЛучевой метод:")
    for i, p in enumerate(points, 1):
        print(f"M{i:02d} = ({p.x:.2f}, {p.y:.2f}) -> {locate_point_ray_method(p, poly)}")


def plot_segments(segments: List[Segment], results: List[Dict[str, Any]], title: str):
    plt.figure(figsize=(10, 8))

    for s in segments:
        plt.plot([s.a.x, s.b.x], [s.a.y, s.b.y], linewidth=2)
        mx = (s.a.x + s.b.x) / 2
        my = (s.a.y + s.b.y) / 2
        plt.text(mx, my, s.name, fontsize=9)

    for item in results:
        if item["type"] == "point":
            p = item["point"]
            plt.scatter([p.x], [p.y], s=60, color="red", zorder=10)

        elif item["type"] == "overlap":
            seg = item["segment"]
            plt.plot(
                [seg.a.x, seg.b.x],
                [seg.a.y, seg.b.y],
                linewidth=6, color="red", alpha=0.75, zorder=11
            )

    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_polygon_and_points(poly: List[Point], points: List[Point], title: str):
    plt.figure(figsize=(10, 8))
    xs = [p.x for p in poly] + [poly[0].x]
    ys = [p.y for p in poly] + [poly[0].y]

    plt.plot(xs, ys, "b-", linewidth=2)
    plt.fill(xs, ys, alpha=0.25)

    for i, p in enumerate(poly, 1):
        plt.text(p.x + 0.1, p.y + 0.1, f"P{i}", color="blue")

    for i, pt in enumerate(points, 1):
        state = locate_point_ray_method(pt, poly)
        if state == "inside":
            color = "green"
        elif state == "boundary":
            color = "red"
        else:
            color = "orange"

        plt.scatter([pt.x], [pt.y], color=color, s=50)
        plt.text(pt.x + 0.1, pt.y + 0.1, f"M{i}", color=color)

    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():
    print("Лабораторная работа №3")
    print("Пересечение отрезков и локализация точки в многоугольнике\n")

    n = int(input("Введите количество отрезков n (n >= 10): "))
    seed_input = input("Введите seed для генерации (или Enter для случайного): ").strip()
    seed = None if seed_input == "" else int(seed_input)

    segments_all = generate_random_segments_for_lr3(n=n, seed=seed)
    segments_non_collinear, segments_with_collinear = split_segments_for_task(segments_all)

    print_segments_list("Сгенерированные отрезки", segments_all)

    # ---------------- ЗАДАНИЕ 1.I ----------------
    res_line = intersections_by_line_equations(segments_non_collinear, skip_collinear=True)
    res_cross = intersections_by_cross_products(segments_non_collinear, allow_collinear=False)
    res_sweep = sweep_line_intersections(segments_non_collinear, allow_collinear=False)

    print_intersections("Задание 1.I.a — по уравнениям прямых", res_line)
    print_intersections("Задание 1.I.b — методом косых произведений", res_cross)
    print_intersections("Задание 1.I.c — методом заметающей прямой", res_sweep)

    plot_segments(segments_non_collinear, res_line, "Задание 1.I.a — уравнения прямых")
    plot_segments(segments_non_collinear, res_cross, "Задание 1.I.b — косые произведения")
    plot_segments(segments_non_collinear, res_sweep, "Задание 1.I.c — заметающая прямая")

    # ---------------- ЗАДАНИЕ 1.II ----------------
    res_cross_col = intersections_by_cross_products(segments_with_collinear, allow_collinear=True)
    res_sweep_col = sweep_line_intersections(segments_with_collinear, allow_collinear=True)

    print_intersections("Задание 1.II.b — косые произведения с коллинеарными", res_cross_col)
    print_intersections("Задание 1.II.c — заметающая прямая с коллинеарными", res_sweep_col)

    plot_segments(segments_with_collinear, res_cross_col, "Задание 1.II.b — косые произведения + коллинеарные")
    plot_segments(segments_with_collinear, res_sweep_col, "Задание 1.II.c — заметающая прямая + коллинеарные")

    # ---------------- ЗАДАНИЕ 2 ----------------
    k = int(input("\nВведите количество точек M_k для многоугольника (k >= 10): "))
    poly = build_nonconvex_polygon()
    poly_mod = build_modified_polygon_with_horizontal_vertices()
    test_points = generate_random_points_for_polygon(k, seed=seed)

    print("\nВершины невыпуклого многоугольника:")
    for i, p in enumerate(poly, 1):
        print(f"P{i} = ({p.x}, {p.y})")

    print_point_locations("Задание 2.I — невыпуклый многоугольник", poly, test_points)
    plot_polygon_and_points(poly, test_points, "Задание 2.I — невыпуклый многоугольник")

    print("\nВершины модифицированного многоугольника:")
    for i, p in enumerate(poly_mod, 1):
        print(f"P{i} = ({p.x}, {p.y})")

    print_point_locations("Задание 2.II — модифицированный многоугольник", poly_mod, test_points)
    plot_polygon_and_points(poly_mod, test_points, "Задание 2.II — модифицированный многоугольник")

    print("\nГотово.")


if __name__ == "__main__":
    main()

