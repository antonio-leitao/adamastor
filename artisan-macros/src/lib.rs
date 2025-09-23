extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Expr, Field, Fields, Ident, ItemFn, ItemStruct, Lit, Meta, ReturnType, Stmt,
    parse_macro_input,
};

#[proc_macro_attribute]
pub fn schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(item as ItemStruct);
    let name = &ast.ident;

    ast.vis = syn::parse_quote!(pub);
    let derives_attr: Attribute =
        syn::parse_quote!(#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]);
    ast.attrs.push(derives_attr);

    // Generate the impl block AND get a cleaned-up version of the fields
    // (with the #[gemini] attributes removed)
    let (gemini_schema_impl, cleaned_fields) =
        generate_gemini_schema_impl_and_clean_fields(&name, &ast.fields);

    // Replace the struct's fields with the cleaned version
    if let Fields::Named(ref mut fields) = ast.fields {
        fields.named = cleaned_fields;
    }

    let output = quote! {
        #ast // This now uses the cleaned-up struct
        #gemini_schema_impl
    };

    output.into()
}

fn generate_gemini_schema_impl_and_clean_fields(
    name: &Ident,
    fields: &Fields,
) -> (
    proc_macro2::TokenStream,
    syn::punctuated::Punctuated<Field, syn::Token![,]>,
) {
    let fields_iter = match fields {
        Fields::Named(fields) => fields.named.iter(),
        _ => panic!("#[schema] can only be used on structs with named fields"),
    };

    let mut cleaned_fields = syn::punctuated::Punctuated::new();

    let properties_quotes = fields_iter.map(|f| {
        let mut cleaned_field = f.clone(); // Clone the field to modify it

        let field_name = f.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_type = &f.ty;

        let description = f.attrs.iter().find_map(|attr| {
            if attr.path().is_ident("gemini") {
                if let Meta::List(meta_list) = &attr.meta {
                    if let Ok(expr) = meta_list.parse_args::<syn::ExprAssign>() {
                        if let syn::Expr::Lit(expr_lit) = *expr.right {
                            if let Lit::Str(lit_str) = expr_lit.lit {
                                return Some(lit_str.value());
                            }
                        }
                    }
                }
            } else if attr.path().is_ident("doc") {
                if let Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let Lit::Str(lit_str) = &expr_lit.lit {
                            return Some(lit_str.value().trim().to_string());
                        }
                    }
                }
            }
            None
        });

        // FIX: Remove the `#[gemini(...)]` attribute from the cleaned field
        cleaned_field
            .attrs
            .retain(|attr| !attr.path().is_ident("gemini"));
        cleaned_fields.push(cleaned_field);

        let desc_quote = if let Some(desc) = description {
            quote! {
                if let Some(obj) = field_schema.as_object_mut() {
                    obj.insert("description".to_string(), serde_json::json!(#desc));
                }
            }
        } else {
            quote! {}
        };

        let type_str = quote!(#field_type).to_string();
        // Remove whitespace to make string matching reliable
        let type_str = quote!(#field_type).to_string().replace(" ", "");

        // --- REORDERED AND IMPROVED LOGIC ---
        let field_schema_type =
            if type_str.contains("Vec<String>") || type_str.contains("Vec<&str>") {
                quote! { serde_json::json!({"type": "ARRAY", "items": {"type": "STRING"}}) }
            } else if type_str.contains("Vec<") {
                // Fallback for other Vec types
                quote! { serde_json::json!({"type": "ARRAY", "items": {"type": "STRING"}}) }
            } else if type_str.contains("Option<") {
                // Check for Option before String
                quote! { serde_json::json!({"type": "STRING", "nullable": true}) }
            } else if type_str.contains("String") {
                // General String check is now later
                quote! { serde_json::json!({"type": "STRING"}) }
            } else if type_str.contains("u32") || type_str.contains("i32") {
                quote! { serde_json::json!({"type": "INTEGER", "format": "int32"}) }
            } else if type_str.contains("u64") || type_str.contains("i64") {
                quote! { serde_json::json!({"type": "INTEGER", "format": "int64"}) }
            } else if type_str.contains("f32") {
                quote! { serde_json::json!({"type": "NUMBER", "format": "float"}) }
            } else if type_str.contains("f64") {
                quote! { serde_json::json!({"type": "NUMBER", "format": "double"}) }
            } else if type_str.contains("bool") {
                quote! { serde_json::json!({"type": "BOOLEAN"}) }
            } else {
                // Default for unknown types
                quote! { serde_json::json!({"type": "STRING"}) }
            };

        quote! {
            {
                let mut field_schema = #field_schema_type;
                #desc_quote
                properties.insert(#field_name_str.to_string(), field_schema);
                required.push(#field_name_str.to_string());
            }
        }
    });

    let impl_block = quote! {
        impl artisan::GeminiSchema for #name {
            fn gemini_schema() -> serde_json::Value {
                let mut properties = serde_json::Map::new();
                let mut required = vec![];

                #(#properties_quotes)*

                serde_json::json!({
                    "type": "OBJECT",
                    "properties": properties,
                    "required": required
                })
            }
        }
    };

    (impl_block, cleaned_fields)
}

#[proc_macro_attribute]
pub fn prompt(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(item as ItemFn);
    let fn_name = &ast.sig.ident;

    let (input_name, input_type, fn_args) = if let Some(syn::FnArg::Typed(pt)) =
        ast.sig.inputs.first()
    {
        let name = if let syn::Pat::Ident(pat_ident) = &*pt.pat {
            pat_ident.ident.to_string()
        } else {
            panic!("#[prompt] argument must be a simple identifier (e.g., `req: RecipeRequest`).")
        };
        let ty = &pt.ty;
        (name, quote! { #ty }, quote! { _: #ty })
    } else {
        // No arguments case
        ("".to_string(), quote! { () }, quote! {})
    };

    let output_type = if let ReturnType::Type(_, ty) = &ast.sig.output {
        ty
    } else {
        panic!("#[prompt] function must have a return type.");
    };

    let template_str = if let Some(Stmt::Expr(Expr::Lit(expr_lit), _)) = ast.block.stmts.first() {
        if let Lit::Str(lit_str) = &expr_lit.lit {
            lit_str.value()
        } else {
            panic!("#[prompt] function body must be a single string literal.");
        }
    } else {
        panic!("#[prompt] function body must be a single string literal.");
    };

    // --- REVISED MACRO OUTPUT ---
    let expanded = quote! {
        // The original function is now just a marker
        #[allow(non_camel_case_types)]
        pub struct #fn_name;

        // Implement our new trait for this marker struct
        impl artisan::IntoPrompt for #fn_name {
            type Input = #input_type;
            type Output = #output_type;

            fn into_prompt(self) -> artisan::Prompt<Self::Input, Self::Output> {
                 artisan::Prompt::new(#template_str, #input_name)
            }
        }
    };

    expanded.into()
}

#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let original_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &original_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let fn_name_inner = format_ident!("__{}_impl", fn_name);

    // Get input type from the function signature
    let input_type = if let Some(syn::FnArg::Typed(pt)) = original_fn.sig.inputs.first() {
        &pt.ty
    } else {
        panic!("#[tool] function must have exactly one argument");
    };

    // Rename the original function to avoid conflicts
    let mut inner_fn = original_fn.clone();
    inner_fn.sig.ident = fn_name_inner.clone();
    inner_fn.vis = syn::Visibility::Inherited; // Make it private

    let expanded = quote! {
        // The actual function implementation (renamed and private)
        #inner_fn

        // Create a unit struct that will act as our tool marker
        #[allow(non_camel_case_types)]
        pub struct #fn_name;

        // Implement IntoTool for this type
        impl artisan::IntoTool for #fn_name {
            fn into_tool(self) -> Box<dyn artisan::GeminiTool> {
                Box::new(artisan::tool(#fn_name_str, #fn_name_inner))
            }
        }
    };

    expanded.into()
}
