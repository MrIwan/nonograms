from solver import create_nng, nng_base

row_res, col_res = create_nng.create_nng_from_image(
    path = 'ressources/Eichhörnchen.jpg',
    rows=2,
    colums=2,
    show_image=True
)

print(f'row_res = {len(row_res)} ')
print(f'col_res = {len(col_res)}')

nng = nng_base.NNGStupidRec(
    row_restrictions=row_res,
    col_restrictions=col_res,
    save_history=True)
nng.solve()