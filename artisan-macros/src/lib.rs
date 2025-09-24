extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Field, Fields, FnArg, Ident, ItemFn, ItemStruct, Lit, Meta, PatIdent, PatType,
    ReturnType, parse_macro_input,
};

#[proc_macro_attribute]
pub fn schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(item as ItemStruct);
    let name = &ast.ident;

    ast.vis = syn::parse_quote!(pub);
    let derives_attr: Attribute =
        syn::parse_quote!(#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]);
    ast.attrs.push(derives_attr);

    let (gemini_schema_impl, cleaned_fields) =
        generate_gemini_schema_impl_and_clean_fields(&name, &ast.fields);

    if let Fields::Named(ref mut fields) = ast.fields {
        fields.named = cleaned_fields;
    }

    let output = quote! {
        #ast
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
        let mut cleaned_field = f.clone();

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

        let type_str = quote!(#field_type).to_string().replace(" ", "");

        let field_schema_type =
            if type_str.contains("Vec<String>") || type_str.contains("Vec<&str>") {
                quote! { serde_json::json!({"type": "ARRAY", "items": {"type": "STRING"}}) }
            } else if type_str.contains("Vec<") {
                quote! { serde_json::json!({"type": "ARRAY", "items": {"type": "STRING"}}) }
            } else if type_str.contains("Option<") {
                quote! { serde_json::json!({"type": "STRING", "nullable": true}) }
            } else if type_str.contains("String") {
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

    // Parse multiple inputs
    let inputs: Vec<(String, Box<syn::Type>)> = ast
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, ty, .. }) = arg {
                if let syn::Pat::Ident(PatIdent { ident, .. }) = &**pat {
                    return Some((ident.to_string(), ty.clone()));
                }
            }
            None
        })
        .collect();

    let output_type = if let ReturnType::Type(_, ty) = &ast.sig.output {
        ty
    } else {
        panic!("#[prompt] function must have a return type.");
    };

    // Extract the function body expression
    let body_expr = &ast.block;

    // Generate wrapper struct if multiple inputs
    let (input_type, wrapper_struct) = if inputs.is_empty() {
        (quote! { () }, quote! {})
    } else if inputs.len() == 1 {
        let ty = &inputs[0].1;
        (quote! { #ty }, quote! {})
    } else {
        let wrapper_name = format_ident!("{}Input", fn_name);

        // Create field definitions for the struct
        let field_defs: Vec<_> = inputs
            .iter()
            .map(|(name, ty)| {
                let ident = format_ident!("{}", name);
                quote! { pub #ident: #ty }
            })
            .collect();

        // Create schema properties
        let schema_props: Vec<_> = inputs
            .iter()
            .map(|(name, ty)| {
                let name_str = name.as_str();
                quote! {
                    properties.insert(#name_str.to_string(),
                        <#ty as artisan::GeminiSchema>::gemini_schema());
                    required.push(#name_str.to_string());
                }
            })
            .collect();

        let wrapper = quote! {
            #[derive(serde::Serialize, serde::Deserialize, Default)]
            pub struct #wrapper_name {
                #(#field_defs),*
            }

            impl artisan::GeminiSchema for #wrapper_name {
                fn gemini_schema() -> serde_json::Value {
                    let mut properties = serde_json::Map::new();
                    let mut required = vec![];

                    #(#schema_props)*

                    serde_json::json!({
                        "type": "OBJECT",
                        "properties": properties,
                        "required": required
                    })
                }
            }
        };

        (quote! { #wrapper_name }, wrapper)
    };

    // Generate the template function
    let template_fn = if inputs.is_empty() {
        quote! {
            Box::new(move |_: &()| -> String {
                #body_expr
            })
        }
    } else if inputs.len() == 1 {
        let input_name = format_ident!("{}", inputs[0].0);
        let input_type_single = &inputs[0].1;
        quote! {
            Box::new(move |#input_name: &#input_type_single| -> String {
                #body_expr
            })
        }
    } else {
        // For multiple inputs, destructure the wrapper
        let destructure: Vec<_> = inputs
            .iter()
            .map(|(name, _)| {
                let ident = format_ident!("{}", name);
                quote! { #ident }
            })
            .collect();

        quote! {
            Box::new(move |input: &#input_type| -> String {
                let #input_type { #(#destructure),* } = input;
                #body_expr
            })
        }
    };

    let expanded = quote! {
        #wrapper_struct

        #[allow(non_camel_case_types)]
        pub struct #fn_name;

        impl artisan::IntoPrompt for #fn_name {
            type Input = #input_type;
            type Output = #output_type;

            fn into_prompt(self) -> artisan::Prompt<Self::Input, Self::Output> {
                artisan::Prompt::new(#template_fn)
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

    // Parse multiple inputs
    let inputs: Vec<(String, Box<syn::Type>)> = original_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, ty, .. }) = arg {
                if let syn::Pat::Ident(PatIdent { ident, .. }) = &**pat {
                    return Some((ident.to_string(), ty.clone()));
                }
            }
            None
        })
        .collect();

    // Generate wrapper struct if multiple inputs
    let (input_type, wrapper_struct, call_expr) = if inputs.is_empty() {
        panic!("#[tool] function must have at least one argument");
    } else if inputs.len() == 1 {
        let ty = &inputs[0].1;
        let arg_name = format_ident!("{}", inputs[0].0);
        (
            quote! { #ty },
            quote! {},
            quote! { #fn_name_inner(#arg_name).await },
        )
    } else {
        let wrapper_name = format_ident!("{}Input", fn_name);

        // Create field definitions for the struct
        let field_defs: Vec<_> = inputs
            .iter()
            .map(|(name, ty)| {
                let ident = format_ident!("{}", name);
                quote! { pub #ident: #ty }
            })
            .collect();

        // Create schema properties
        let schema_props: Vec<_> = inputs
            .iter()
            .map(|(name, ty)| {
                let name_str = name.as_str();
                quote! {
                    properties.insert(#name_str.to_string(),
                        <#ty as artisan::GeminiSchema>::gemini_schema());
                    required.push(#name_str.to_string());
                }
            })
            .collect();

        // Create field names for destructuring
        let field_names: Vec<_> = inputs
            .iter()
            .map(|(name, _)| {
                let ident = format_ident!("{}", name);
                quote! { #ident }
            })
            .collect();

        let wrapper = quote! {
            #[derive(serde::Serialize, serde::Deserialize)]
            pub struct #wrapper_name {
                #(#field_defs),*
            }

            impl artisan::GeminiSchema for #wrapper_name {
                fn gemini_schema() -> serde_json::Value {
                    let mut properties = serde_json::Map::new();
                    let mut required = vec![];

                    #(#schema_props)*

                    serde_json::json!({
                        "type": "OBJECT",
                        "properties": properties,
                        "required": required
                    })
                }
            }
        };

        let call = quote! {
            #fn_name_inner(#(input.#field_names),*).await
        };

        (quote! { #wrapper_name }, wrapper, call)
    };

    // Clone and modify the inner function
    let mut inner_fn = original_fn.clone();
    inner_fn.sig.ident = fn_name_inner.clone();
    inner_fn.vis = syn::Visibility::Inherited;

    let expanded = quote! {
        #wrapper_struct

        #inner_fn

        #[allow(non_camel_case_types)]
        pub struct #fn_name;

        impl artisan::IntoTool for #fn_name {
            fn into_tool(self) -> Box<dyn artisan::GeminiTool> {
                Box::new(artisan::tool(#fn_name_str, |input: #input_type| async move {
                    #call_expr
                }))
            }
        }
    };

    expanded.into()
}
