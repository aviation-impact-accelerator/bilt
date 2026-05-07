"""mypy plugin to make mypy compatible with Bilt."""

__all__ = ["plugin"]

import typing

import mypy.nodes
import mypy.plugin
import mypy.plugins.common
import mypy.plugins.dataclasses
import mypy.types

from bilt.core._system._decorator import system

SELF_TYPE_VAR_NAME: typing.Final = "_DT"

SYSTEM_DECORATOR_FULLNAME: typing.Final = f"{system.__module__}.{system.__name__}"


def plugin(version: str) -> type[mypy.plugin.Plugin]:  # noqa: ARG001
    """Entry point provided to mypy for accessing `BiltMyPyPlugin."""
    return BiltMyPyPlugin


class BiltMyPyPlugin(mypy.plugin.Plugin):
    """mypy plugin to make mypy compatible with Bilt."""

    def get_class_decorator_hook(self, fullname: str) -> typing.Callable[[mypy.plugin.ClassDefContext], None] | None:
        if fullname == SYSTEM_DECORATOR_FULLNAME:
            return mypy.plugins.dataclasses.dataclass_tag_callback
        return None

    def get_class_decorator_hook_2(self, fullname: str) -> typing.Callable[[mypy.plugin.ClassDefContext], bool] | None:
        if fullname == SYSTEM_DECORATOR_FULLNAME:
            return _bilt_system_maker_callback
        return None


def _bilt_system_maker_callback(ctx: mypy.plugin.ClassDefContext) -> bool:
    # Determine the fields from the `system` definition. The `system` is defined like a dataclass, so we can use
    # the dataclass plugin to extract the fields.
    transformer = mypy.plugins.dataclasses.DataclassTransformer(
        ctx.cls,
        ctx.reason,
        mypy.plugins.dataclasses._TRANSFORM_SPEC_FOR_DATACLASSES,  # noqa: SLF001
        ctx.api,
    )
    transformer.transform()

    # `transformer.transform` will exit early and return `False` if `attributes is None`, so we
    # know at this point that mypy the attribute definitions are ready and we can cast `attributes`
    # to `list[mypy.plugin.dataclasses.DataclassAttribute]` to narrow its type from a union with
    # `None`.
    fields = typing.cast("list[mypy.plugins.dataclasses.DataclassAttribute]", transformer.collect_attributes())

    _bilt_system_set_interface_metaclass(ctx)
    _bilt_system_add_init_method(ctx, fields)

    return True


def _bilt_system_set_interface_metaclass(ctx: mypy.plugin.ClassDefContext) -> None:
    interface_symbol = ctx.api.lookup_fully_qualified("bilt.core._system._interface.Interface")
    metaclass_info = typing.cast("mypy.nodes.TypeInfo", interface_symbol.node)
    ctx.cls.info.metaclass_type = mypy.types.Instance(metaclass_info, [])


def _bilt_system_add_init_method(
    ctx: mypy.plugin.ClassDefContext, fields: list[mypy.plugins.dataclasses.DataclassAttribute]
) -> None:
    method_name = "__init__"
    type_var_def = _construct_type_var_def(ctx, method_name)

    missing_symbol = ctx.api.lookup_fully_qualified("bilt.core._system._interface.MISSING")
    missing_ref = mypy.nodes.NameExpr("MISSING")
    missing_ref.node = missing_symbol.node
    # `missing_symbol.fullname` cannot be `None` because `MISSING` is always defined.
    missing_ref.fullname = missing_symbol.fullname  # type: ignore[assignment]
    missing_ref.kind = missing_symbol.kind
    missing_type = ctx.api.named_type("bilt.core._system._interface._Missing")

    arguments = []
    for field in fields:
        # `field.type` cannot be `None` because this is invalid for a `system` definition. We need to validate that this
        # is actually the case (and error if not) earlier on in the process.
        field.type = typing.cast("mypy.types.Type", field.type)
        arguments.append(
            mypy.nodes.Argument(
                mypy.nodes.Var(
                    field.name,
                    mypy.types.UnionType([field.type, missing_type]),
                ),
                mypy.types.UnionType([field.type, missing_type]),
                missing_ref,
                mypy.nodes.ARG_OPT,
            )
        )
    mypy.plugins.common.add_method_to_class(
        ctx.api,
        ctx.cls,
        method_name,
        args=arguments,
        return_type=mypy.types.NoneType(),
        self_type=type_var_def,
        tvar_def=type_var_def,
    )


def _construct_type_var_def(ctx: mypy.plugin.ClassDefContext, method_name: str) -> mypy.types.TypeVarType:
    return mypy.types.TypeVarType(
        SELF_TYPE_VAR_NAME,
        f"{ctx.cls.info.fullname}.{SELF_TYPE_VAR_NAME}",
        id=mypy.types.TypeVarId(-1, namespace=f"{ctx.cls.info.fullname}.{method_name}"),
        values=[],
        upper_bound=ctx.api.named_type("builtins.object"),
        default=mypy.types.AnyType(mypy.types.TypeOfAny.from_omitted_generics),
    )
