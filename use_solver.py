from solver import create_nng, nng_base, nng_shift_register

row_res, col_res = create_nng.create_nng_from_image(
    path = 'ressources/Cats.jpg',
    rows=4,
    colums=4,
    show_image=True
)


nng = nng_shift_register.NNGShitRegister(
    row_restrictions=row_res,
    col_restrictions=col_res,
    save_history=True)
nng.solve()