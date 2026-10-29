# Interactions

An interaction identifies the physical law and its immutable scalar
parameters. Model specifications expose an interaction, and result comparison
checks the pair `(name, parameters)` when deciding whether two results share
that interpretation.

`gragra.interactions` exports the interaction protocol and one built-in
parameter-free interaction. These names are not re-exported from the `gragra`
root package.

::: gragra.interactions.Interaction

`Newtonian` identifies the parameter-free Newtonian gravitational interaction.
Its stable identity is the name `"newton"` and an empty immutable parameter
mapping.

::: gragra.interactions.Newtonian
