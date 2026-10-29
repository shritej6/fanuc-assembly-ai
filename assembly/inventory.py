"""Observed inventory for one-at-a-time recipe ownership.

Only perception-facing instance and identity data enter this module. Simulator
ground-truth poses or privileged object state are neither accepted nor stored.
"""

from dataclasses import dataclass, replace
from enum import StrEnum

from assembly.catalog import Catalog, UnknownBaseError, UnknownProductError


class InventoryError(Exception):
    """Base class for typed live-inventory failures."""


class DuplicateInstanceError(InventoryError):
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        super().__init__(f"Inventory instance {instance_id!r} already exists")


class UnknownInventoryItemError(InventoryError):
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        super().__init__(f"Unknown inventory instance {instance_id!r}")


class InvalidInventoryObjectError(InventoryError):
    def __init__(self, object_id: object) -> None:
        self.object_id = object_id
        super().__init__(f"Object {object_id!r} is not present in the validated catalog")


class InvalidInventoryProductError(InventoryError):
    def __init__(self, product: object) -> None:
        self.product = product
        super().__init__(f"Unknown inventory product {product!r}")


class InvalidInventoryObservationError(InventoryError):
    """Raised when observed identity data is malformed."""


class InvalidInventoryMoveError(InventoryError):
    """Raised when a zone change violates live-inventory ownership."""


class InvalidObjectConditionError(InventoryError):
    def __init__(self, condition: object) -> None:
        self.condition = condition
        super().__init__(f"Unknown object condition {condition!r}")


class InventoryOwnershipError(InventoryError):
    def __init__(self, owner: int, requested: int) -> None:
        self.owner = owner
        self.requested = requested
        super().__init__(f"Product {owner} owns inventory reservations; product {requested} must wait")


class NoActiveReservationError(InventoryError):
    """Raised when a reservation-only operation has no current owner."""


class InventoryShortageError(InventoryError):
    """A missing exact recipe item that should suspend and raise an Andon."""

    recommended_state = "Suspended"
    andon_required = True

    def __init__(
        self,
        product: int,
        *,
        required_object_id: str | None = None,
        required_base_type: str | None = None,
    ) -> None:
        self.product = product
        self.required_object_id = required_object_id
        self.required_base_type = required_base_type
        required = required_object_id or f"base type {required_base_type}"
        super().__init__(f"Product {product} requires verified {required}; no usable table part or spare exists")


class ReservationPartError(InventoryError):
    def __init__(self, product: int, object_id: str) -> None:
        self.product = product
        self.object_id = object_id
        super().__init__(f"Product {product} has no reservation for {object_id}")


class IncompleteProductError(InventoryError):
    def __init__(self, product: int, instance_ids: tuple[str, ...]) -> None:
        self.product = product
        self.instance_ids = instance_ids
        joined = ", ".join(instance_ids)
        super().__init__(f"Product {product} cannot complete; unsafe inventory instances: {joined}")


class InventoryZone(StrEnum):
    TABLE = "table"
    SPARE = "spare"
    RESERVED = "reserved"
    INSTALLED = "installed"
    STAGED = "staged"
    OUTPUT = "output"
    REJECT = "reject"


class ObjectCondition(StrEnum):
    VERIFIED = "verified"
    FALLEN = "fallen"
    DEFECTIVE = "defective"
    UNCERTAIN = "uncertain"
    NOT_OBSERVED = "not_observed"


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """A perception-verified object instance and its live inventory state."""

    instance_id: str
    object_id: str
    zone: InventoryZone
    condition: ObjectCondition
    reserved_for: int | None = None
    source_zone: InventoryZone | None = None


@dataclass(frozen=True, slots=True)
class ProductReservation:
    """One compatible fixture and the four exact recipe parts."""

    product: int
    base: InventoryItem
    parts: tuple[InventoryItem, ...]

    @property
    def items(self) -> tuple[InventoryItem, ...]:
        return (self.base, *self.parts)

    def part(self, object_id: str) -> InventoryItem:
        for item in self.parts:
            if item.object_id == object_id:
                return item
        raise ReservationPartError(self.product, object_id)


class Inventory:
    """Live zones and atomic reservations backed by a validated catalog."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._items: dict[str, InventoryItem] = {}
        self._part_ids = {step.object_id for product in catalog.products for step in product.steps}
        self._active: ProductReservation | None = None

    @property
    def active_product(self) -> int | None:
        return None if self._active is None else self._active.product

    @property
    def active_reservation(self) -> ProductReservation:
        if self._active is None:
            raise NoActiveReservationError("No product owns an active reservation")
        return self._active

    def add(
        self,
        instance_id: str,
        object_id: str,
        *,
        zone: InventoryZone | str = InventoryZone.TABLE,
        condition: ObjectCondition | str = ObjectCondition.VERIFIED,
    ) -> InventoryItem:
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise InvalidInventoryObservationError("instance_id must be a non-empty observed identifier")
        if instance_id in self._items:
            raise DuplicateInstanceError(instance_id)
        self._validate_object(object_id)
        normalized_zone = self._normalize_zone(zone)
        if normalized_zone in {
            InventoryZone.RESERVED,
            InventoryZone.INSTALLED,
            InventoryZone.STAGED,
            InventoryZone.OUTPUT,
        }:
            raise InvalidInventoryMoveError(f"Cannot observe a new item directly in {normalized_zone.value}")
        item = InventoryItem(instance_id, object_id, normalized_zone, self._normalize_condition(condition))
        self._items[instance_id] = item
        return item

    def item(self, instance_id: str) -> InventoryItem:
        try:
            return self._items[instance_id]
        except KeyError as error:
            raise UnknownInventoryItemError(instance_id) from error

    def items(self, zone: InventoryZone | str | None = None) -> tuple[InventoryItem, ...]:
        items = tuple(self._items.values())
        if zone is None:
            return items
        normalized = self._normalize_zone(zone)
        return tuple(item for item in items if item.zone is normalized)

    def set_condition(
        self,
        instance_id: str,
        condition: ObjectCondition | str,
    ) -> InventoryItem:
        item = self.item(instance_id)
        normalized = self._normalize_condition(condition)
        if (
            item.zone in {InventoryZone.RESERVED, InventoryZone.INSTALLED, InventoryZone.STAGED}
            and normalized is not ObjectCondition.VERIFIED
        ):
            raise InvalidInventoryMoveError(f"Reject {instance_id!r} before marking a claimed item unusable")
        updated = replace(item, condition=normalized)
        self._items[instance_id] = updated
        return updated

    def reconcile_observation(
        self,
        instance_id: str,
        object_id: str,
        *,
        zone: InventoryZone | str = InventoryZone.TABLE,
        condition: ObjectCondition | str = ObjectCondition.VERIFIED,
    ) -> InventoryItem:
        """Apply a fresh perception result without discarding cell ownership."""
        current = self.item(instance_id)
        if current.object_id != object_id:
            raise InvalidInventoryObservationError(
                f"Inventory instance {instance_id!r} changed identity from {current.object_id!r} to {object_id!r}"
            )
        observed_zone = self._normalize_observed_zone(zone)
        observed_condition = self._normalize_condition(condition)
        if current.zone in {InventoryZone.TABLE, InventoryZone.SPARE}:
            updated = replace(
                current,
                zone=observed_zone,
                condition=observed_condition,
                reserved_for=None,
                source_zone=None,
            )
        elif current.zone in {InventoryZone.RESERVED, InventoryZone.INSTALLED, InventoryZone.STAGED}:
            updated = replace(current, condition=observed_condition)
        else:
            updated = current
        self._items[instance_id] = updated
        self._refresh_active_item(updated)
        return updated

    def mark_unobserved(self, instance_id: str) -> InventoryItem:
        """Make an expected-but-absent instance unusable until it is seen again."""
        current = self.item(instance_id)
        updated = replace(current, condition=ObjectCondition.NOT_OBSERVED)
        self._items[instance_id] = updated
        self._refresh_active_item(updated)
        return updated

    def reserve_product(self, product: int) -> ProductReservation:
        recipe = self._recipe(product)
        if self._active is not None:
            if self._active.product != product:
                raise InventoryOwnershipError(self._active.product, product)
            return self._active
        base = self._find_base(recipe.base_type)
        if base is None:
            raise InventoryShortageError(product, required_base_type=recipe.base_type)
        parts = tuple(self._require_part(product, step.object_id) for step in recipe.steps)
        reserved = tuple(self._reserve(item, product) for item in (base, *parts))
        self._active = ProductReservation(product, reserved[0], reserved[1:])
        return self._active

    def install(self, instance_id: str) -> InventoryItem:
        item = self.item(instance_id)
        if item.zone is not InventoryZone.RESERVED or item.reserved_for != self.active_product:
            raise InvalidInventoryMoveError(f"Instance {instance_id!r} is not reserved by the active product")
        updated = replace(item, zone=InventoryZone.INSTALLED, source_zone=None)
        self._items[instance_id] = updated
        self._refresh_active_item(updated)
        return updated

    def stage_for_rebuild(self, instance_id: str) -> InventoryItem:
        item = self.item(instance_id)
        if item.zone is not InventoryZone.INSTALLED or item.reserved_for != self.active_product:
            raise InvalidInventoryMoveError(f"Instance {instance_id!r} is not an installed active-product part")
        updated = replace(item, zone=InventoryZone.STAGED)
        self._items[instance_id] = updated
        self._refresh_active_item(updated)
        return updated

    def reinstall(self, instance_id: str) -> InventoryItem:
        item = self.item(instance_id)
        if item.zone is not InventoryZone.STAGED or item.reserved_for != self.active_product:
            raise InvalidInventoryMoveError(f"Instance {instance_id!r} is not staged by the active product")
        updated = replace(item, zone=InventoryZone.INSTALLED)
        self._items[instance_id] = updated
        self._refresh_active_item(updated)
        return updated

    def reject(
        self,
        instance_id: str,
        *,
        condition: ObjectCondition | str = ObjectCondition.DEFECTIVE,
    ) -> InventoryItem:
        item = self.item(instance_id)
        updated = replace(
            item,
            zone=InventoryZone.REJECT,
            condition=self._normalize_condition(condition),
            reserved_for=None,
            source_zone=None,
        )
        self._items[instance_id] = updated
        self._refresh_active_item(updated)
        return updated

    def replace_with_spare(
        self,
        instance_id: str,
        *,
        condition: ObjectCondition | str = ObjectCondition.DEFECTIVE,
    ) -> InventoryItem:
        active = self.active_reservation
        original = self.item(instance_id)
        if not any(item.instance_id == instance_id for item in active.parts):
            raise InvalidInventoryMoveError(f"Instance {instance_id!r} is not a reserved recipe part")
        if original.zone is InventoryZone.RESERVED:
            if original.reserved_for != active.product:
                raise InvalidInventoryMoveError(f"Instance {instance_id!r} belongs to another product")
            original = self.reject(instance_id, condition=condition)
        elif original.zone is not InventoryZone.REJECT:
            raise InvalidInventoryMoveError(f"Instance {instance_id!r} must be reserved or rejected")
        replacement = self._find_replacement(original.object_id)
        if replacement is None:
            raise InventoryShortageError(active.product, required_object_id=original.object_id)
        reserved = self._reserve(replacement, active.product)
        current = self.active_reservation
        parts = tuple(reserved if item.instance_id == instance_id else item for item in current.parts)
        self._active = replace(current, parts=parts)
        return reserved

    def complete_product(self, product: int) -> ProductReservation:
        """Consume a verified assembled base so it cannot be selected again."""
        active = self.active_reservation
        if active.product != product:
            raise InventoryOwnershipError(active.product, product)
        unsafe = self._completion_failures(active)
        if unsafe:
            raise IncompleteProductError(product, unsafe)
        base = replace(
            active.base,
            zone=InventoryZone.OUTPUT,
            reserved_for=None,
            source_zone=None,
        )
        parts = tuple(replace(part, reserved_for=None, source_zone=None) for part in active.parts)
        completed = ProductReservation(product, base, parts)
        self._items[base.instance_id] = base
        for part in parts:
            self._items[part.instance_id] = part
        self._active = None
        return completed

    def release_reservations(self, product: int | None = None) -> tuple[str, ...]:
        if self._active is None:
            return ()
        owner = self._active.product
        if product is not None and product != owner:
            raise InventoryOwnershipError(owner, product)
        if any(item.zone is InventoryZone.STAGED for item in self._active.parts):
            raise InvalidInventoryMoveError("staged recovery parts require operator reconciliation before reset")
        released = tuple(item.instance_id for item in self._active.items if self._release_item(item.instance_id, owner))
        self._active = None
        return released

    def _validate_object(self, object_id: object) -> None:
        if not isinstance(object_id, str):
            raise InvalidInventoryObjectError(object_id)
        if object_id in self._part_ids:
            return
        try:
            self._catalog.base_type_for_object(object_id)
        except UnknownBaseError as error:
            raise InvalidInventoryObjectError(object_id) from error

    def _recipe(self, product: object):
        if isinstance(product, bool) or not isinstance(product, int):
            raise InvalidInventoryProductError(product)
        try:
            return self._catalog.product(product)
        except UnknownProductError as error:
            raise InvalidInventoryProductError(product) from error

    def _find_base(self, base_type: str) -> InventoryItem | None:
        for item in self._items.values():
            if not self._is_usable(item, InventoryZone.TABLE):
                continue
            try:
                if self._catalog.base_type_for_object(item.object_id) == base_type:
                    return item
            except UnknownBaseError:
                continue
        return None

    @staticmethod
    def _completion_failures(active: ProductReservation) -> tuple[str, ...]:
        unsafe: list[str] = []
        if active.base.zone is not InventoryZone.RESERVED:
            unsafe.append(active.base.instance_id)
        if active.base.condition is not ObjectCondition.VERIFIED:
            unsafe.append(active.base.instance_id)
        unsafe.extend(
            part.instance_id
            for part in active.parts
            if part.zone is not InventoryZone.INSTALLED or part.condition is not ObjectCondition.VERIFIED
        )
        return tuple(dict.fromkeys(unsafe))

    def _require_part(self, product: int, object_id: str) -> InventoryItem:
        for zone in (InventoryZone.TABLE, InventoryZone.SPARE):
            for item in self._items.values():
                if item.object_id == object_id and self._is_usable(item, zone):
                    return item
        raise InventoryShortageError(product, required_object_id=object_id)

    def _find_replacement(self, object_id: str) -> InventoryItem | None:
        for zone in (InventoryZone.TABLE, InventoryZone.SPARE):
            for item in self._items.values():
                if item.object_id == object_id and self._is_usable(item, zone):
                    return item
        return None

    @staticmethod
    def _is_usable(item: InventoryItem, zone: InventoryZone) -> bool:
        return item.zone is zone and item.condition is ObjectCondition.VERIFIED

    def _reserve(self, item: InventoryItem, product: int) -> InventoryItem:
        reserved = replace(
            item,
            zone=InventoryZone.RESERVED,
            reserved_for=product,
            source_zone=item.zone,
        )
        self._items[item.instance_id] = reserved
        return reserved

    def _release_item(self, instance_id: str, owner: int) -> bool:
        current = self.item(instance_id)
        if current.zone is not InventoryZone.RESERVED or current.reserved_for != owner:
            return False
        if current.source_zone not in {InventoryZone.TABLE, InventoryZone.SPARE}:
            raise InvalidInventoryMoveError(f"Reserved instance {instance_id!r} has no safe source zone")
        self._items[instance_id] = replace(
            current,
            zone=current.source_zone,
            reserved_for=None,
            source_zone=None,
        )
        return True

    def _refresh_active_item(self, updated: InventoryItem) -> None:
        if self._active is None:
            return
        base = self._active.base
        if base.instance_id == updated.instance_id:
            self._active = replace(self._active, base=updated)
            return
        parts = tuple(updated if item.instance_id == updated.instance_id else item for item in self._active.parts)
        self._active = replace(self._active, parts=parts)

    @staticmethod
    def _normalize_zone(zone: InventoryZone | str) -> InventoryZone:
        try:
            return InventoryZone(zone)
        except (TypeError, ValueError) as error:
            raise InvalidInventoryMoveError(f"Unknown inventory zone {zone!r}") from error

    @classmethod
    def _normalize_observed_zone(cls, zone: InventoryZone | str) -> InventoryZone:
        normalized = cls._normalize_zone(zone)
        if normalized not in {InventoryZone.TABLE, InventoryZone.SPARE}:
            raise InvalidInventoryMoveError(f"Perception cannot assign internal inventory zone {normalized.value!r}")
        return normalized

    @staticmethod
    def _normalize_condition(condition: ObjectCondition | str) -> ObjectCondition:
        try:
            return ObjectCondition(condition)
        except (TypeError, ValueError) as error:
            raise InvalidObjectConditionError(condition) from error
